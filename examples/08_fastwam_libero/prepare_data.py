"""Convert the published LeRobot episodes to VLAStudio HDF5 without image re-encoding."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path

import av
import h5py
import numpy as np
import pyarrow.parquet as pq

SUITES = ("spatial", "object", "goal", "10")
REVISION = "ee018b997c430bb12b5bf3c892d744798c5a2f91"
ARCHIVE_SHA256 = {
    "spatial": "7c374be75133e1a6779a0e07b9452c16294777be577b89553f1cf5350eafd8f9",
    "object": "732405d5deaf3019c244bacd3f26c9617f25c871750586c9968747b6cfefedda",
    "goal": "a21ae10171535585fb43e6405d9efa09ff38ef34689e4176428ca005af3a39ea",
    "10": "eb5af5f46d9348f8ba0146cf90de718ce102f261348983c0fd2df08d50dd1059",
}


def download_source(source_root, suite):
    import tarfile
    from huggingface_hub import hf_hub_download
    name = f"libero_{suite}_no_noops_lerobot.tar.gz"
    archive = Path(hf_hub_download("yuanty/LIBERO-fastwam", name, repo_type="dataset", revision=REVISION))
    digest = hashlib.sha256()
    with archive.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != ARCHIVE_SHA256[suite]:
        raise ValueError(f"Archive checksum mismatch: {name}")
    source_root = source_root.resolve()
    source_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            path = (source_root / member.name).resolve()
            if source_root not in path.parents or not (member.isfile() or member.isdir()):
                raise ValueError(f"Unsafe archive member: {member.name}")
        tar.extractall(source_root)



def convert_episode(job):
    source, output, suite, ep, info, tasks = job
    source, output = Path(source), Path(output)
    episode = int(ep["episode_index"])
    chunk = episode // int(info.get("chunks_size", 1000))
    parquet = source / info["data_path"].format(episode_chunk=chunk, episode_index=episode)
    fingerprint = hashlib.sha256(parquet.read_bytes()).hexdigest()
    relative = f"{suite}/episode_{episode:06d}.hdf5"
    destination = output / relative
    result = dict(path=relative, length=int(ep["length"]), suite=suite, source_episode=episode,
                  parquet_sha256=fingerprint)
    if destination.exists():
        with h5py.File(destination, "r") as f:
            if (f.attrs.get("parquet_sha256") != fingerprint or not f.attrs.get("complete")
                    or len(f["action_ee"]) != result["length"]):
                raise ValueError(f"Existing episode does not match its source: {destination}")
        return result
    rows = pq.read_table(parquet).to_pydict()
    actions = np.asarray(rows["action"], dtype=np.float32)
    states = np.asarray(rows["observation.state"], dtype=np.float32)
    timestamps = np.asarray(rows["timestamp"], dtype=np.float64)
    if actions.shape != (result["length"], 7) or states.shape != (result["length"], 8):
        raise ValueError(f"Unexpected action/state shape in {parquet}")
    if not np.isfinite(actions).all() or not np.isfinite(states).all():
        raise ValueError(f"Nonfinite values in {parquet}")
    if (actions[:, -1].min() < 0) or (actions[:, -1].max() > 1):
        raise ValueError("Expected the standard gripper convention: 0=closed, 1=open")
    task_ids = set(rows["task_index"])
    if len(task_ids) != 1:
        raise ValueError(f"Expected one task per episode: {parquet}")
    language = tasks[int(next(iter(task_ids)))]
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".partial")
    with h5py.File(temporary, "w") as f:
        f.attrs["parquet_sha256"] = fingerprint
        f.attrs["source"] = str(parquet)
        for key, value in {"dataset_dir": str(output), "episode_id": f"{suite}/{episode}",
                           "language_instruction": language, "robot": "panda"}.items():
            f.create_dataset(key, data=[value], dtype=h5py.string_dtype("utf-8"))
        f.create_dataset("freq", data=[float(info["fps"])])
        f.create_dataset("action_ee", data=actions)
        f.create_dataset("observations/state_ee", data=states)
        f.create_dataset("timestamp", data=timestamps)
        for source_key, camera in (("observation.images.image", "primary"),
                                   ("observation.images.wrist_image", "wrist")):
            video = source / info["video_path"].format(episode_chunk=chunk, episode_index=episode, video_key=source_key)
            # Decode once in timestamp order. Store lossless uint8 pixels at source resolution.
            with av.open(str(video)) as container:
                container.streams.video[0].thread_type = "NONE"
                frames = [(float(frame.time), frame.to_ndarray(format="rgb24")) for frame in container.decode(video=0)]
            if not frames:
                raise ValueError(f"Empty video: {video}")
            frame_times = np.array([t for t, _ in frames])
            matches = np.abs(timestamps[:, None] - frame_times[None]).argmin(axis=1)
            errors = np.abs(frame_times[matches] - timestamps)
            if errors.max() > 1 / float(info["fps"]) / 2:
                raise ValueError(f"Video timestamps do not match parquet: {video}")
            shape = frames[0][1].shape
            target = f.create_dataset(f"observations/image/{camera}", shape=(len(actions), *shape),
                                      dtype="uint8", chunks=(1, *shape), compression="lzf")
            for i, match in enumerate(matches):
                target[i] = frames[match][1]
        f.attrs["complete"] = True
    temporary.replace(destination)
    return result


def main():
    cache = Path(os.environ.get("VLASTUDIO_CACHE", "~/.cache/vlastudio")).expanduser()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=cache / "data/fastwam/libero_mujoco3.3.2")
    p.add_argument("--output", type=Path, default=cache / "data/fastwam/libero_hdf5")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, help="Convert only this many episodes per suite for a smoke test")
    p.add_argument("--text-device", choices=["cpu", "cuda"], help="Also build text embeddings from the converted data")
    args = p.parse_args()
    existing = args.output / "manifest.json"
    if existing.exists() and json.loads(existing.read_text())["partial"] != (args.limit is not None):
        raise ValueError("Use a separate output directory for partial smoke datasets")
    jobs, source_info = [], {}
    for suite in SUITES:
        source = args.source / f"libero_{suite}_no_noops_lerobot"
        if not source.exists():
            download_source(args.source, suite)
        info = json.loads((source / "meta/info.json").read_text())
        episodes = [json.loads(line) for line in (source / "meta/episodes.jsonl").read_text().splitlines() if line]
        tasks = {int(t["task_index"]): t["task"] for t in
                 (json.loads(line) for line in (source / "meta/tasks.jsonl").read_text().splitlines() if line)}
        if info["codebase_version"] != "v2.1" or info["fps"] != 20:
            raise ValueError("Expected the published LeRobot v2.1, 20 FPS export")
        source_info[suite] = dict(episodes=info["total_episodes"], frames=info["total_frames"])
        for ep in sorted(episodes, key=lambda e: e["episode_index"])[:args.limit]:
            jobs.append((str(source), str(args.output), suite, ep, info, tasks))
    args.output.mkdir(parents=True, exist_ok=True)
    episodes = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, result in enumerate(pool.map(convert_episode, jobs)):
            episodes.append(result)
            if (i + 1) % 20 == 0 or i + 1 == len(jobs):
                print(f"Converted {i+1}/{len(jobs)} episodes", flush=True)
    # Fit global statistics directly on the converted training episodes.
    stats = {}
    for kind, field in (("action", "action_ee"), ("state", "observations/state_ee")):
        values = []
        for ep in episodes:
            with h5py.File(args.output / ep["path"], "r") as f:
                values.append(f[field][()])
        values = np.concatenate(values)
        stats[kind] = {"default": {"global_min": values.min(0).tolist(), "global_max": values.max(0).tolist()}}
    (args.output / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    manifest = dict(format="vlastudio-hdf5-v1", complete=True, fps=20,
                    source="yuanty/LIBERO-fastwam", source_revision="ee018b997c430bb12b5bf3c892d744798c5a2f91",
                    source_info=source_info, partial=args.limit is not None,
                    total_frames=sum(e["length"] for e in episodes), episodes=episodes)
    temporary = args.output / "manifest.partial.json"
    temporary.write_text(json.dumps(manifest, indent=2))
    temporary.replace(args.output / "manifest.json")
    if args.text_device:
        os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(cache / "models/fastwam"))
        from vlastudio.policy.fastwam.processing import prepare_text_cache
        prepare_text_cache(args.output, cache / "data/fastwam/text_embeds_cache/libero", args.text_device)
    print(json.dumps({"output": str(args.output), "episodes": len(episodes), "frames": manifest["total_frames"]}))


if __name__ == "__main__":
    main()
