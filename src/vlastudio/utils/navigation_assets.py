"""Verified navigation assets, fetched lazily by datasets and policy loaders."""
import hashlib
import io
import json
import os
from pathlib import Path
import zipfile


def cache_root():
    return Path(os.environ.get('VLASTUDIO_CACHE', '~/.cache/vlastudio')).expanduser()/'navigation'


def dataset_root():
    value = os.environ.get('VLASTUDIO_DATA_CACHE_DIR')
    return Path(value).expanduser()/'navigation' if value else cache_root()/'datasets'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url, path, checksum, download=True):
    """Atomically cache a public asset; local mode never makes a network request."""
    from filelock import FileLock
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path)+'.lock'):
        if path.is_file():
            if checksum and sha256(path) != checksum:
                raise ValueError(f'Asset checksum mismatch: {path}')
            return path
        if not download:
            raise FileNotFoundError(f'Asset missing with download=false: {path}')
        import requests
        temporary = path.with_name(path.name+'.partial')
        with requests.get(url, stream=True, timeout=(20, 180)) as response:
            response.raise_for_status()
            if 'text/html' in response.headers.get('Content-Type', ''):
                raise ValueError(f'Expected an asset, received HTML: {path.name}')
            with temporary.open('wb') as output:
                for chunk in response.iter_content(8*1024*1024):
                    output.write(chunk)
        if checksum and sha256(temporary) != checksum:
            temporary.unlink()
            raise ValueError(f'Download checksum mismatch: {path.name}')
        temporary.replace(path)
    return path


R2R_ASSETS = {
    'train': ('hh5qec8o5urcztn', '30af70d96ea217778ba0d1b3085a774a13f36266fef39a13dbe8b03516fbf988'),
    'val_unseen': ('p6hlckr70a07wka', '1cc730adc1c098998ac9191462849fc725e0c5336984a35bfbac7d5f044e682c'),
    'val_seen': ('8ye4gqce7v8yzdm', '755633db9d5e64d5ca71d329c5046bb722781a09f0d1d501d7e8d094c83652ec'),
}
VLNCE_SHA = '3171f5bed90c81d7eb4db9554955b6229ec1fe96d808c9340582ba33eafdb4de'
CMA_SHA = '06278fdbb52ad8c453b404dacb079fb353252056314b1c567a381df77e6827a3'
BERT_SHA = '612fccc817d104ef09de4df70c081ddd692cbb85c219785c7bbfc627507c3ee0'
VOCAB_SHA = '07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3'
FEATURE_SHA = '791344c4987de20ff792f80159a7f9e229310f421f59e4494c564b938c5bab8d'
R2R_MIRROR_REVISION = '6fe3b2d84de51c6e788fdc361d4f160728228a79'
# JSONL file hash and semantic hash of the complete original ordered records.
R2R_MIRROR = {
    'train': ('e433d1ff5f139bc3ac10af6a2796c7e97d7973d1c6bccda3a79a609d2b83ee75',
              'd548fc336babe3703737f22a3fceb71ec018d93ad40e5057b3331d9be53cc81d'),
    'val_seen': ('85346c644497a30f4b91aecd28c40ad81fac6d110c971d53ee379bd60d648fdc',
                 '6c4d2127e4452bf5b9bd22b263d34d8d6260cbfbbadeaf2868d20615a7bfab1d'),
    'val_unseen': ('68a4e427b7727e03fc4ae57ae32bf11f711d1f5c23bfdcccc34f6ececb69c2e6',
                   '6174294ae731ef46ef697437ae8ff7698fd62d5d938ac4a6a10aac52bfc66a65'),
}


def canonical_json_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()


def r2r_annotations(split='train', root=None, download=True):
    from filelock import FileLock
    import requests
    if split not in R2R_ASSETS:
        raise ValueError('Supported R2R splits: train, val_seen, val_unseen')
    root = Path(root) if root else dataset_root()/'r2r'
    token, checksum = R2R_ASSETS[split]
    root.mkdir(parents=True, exist_ok=True)
    target = root/f'R2R_{split}.json'
    with FileLock(str(target)+'.annotation.lock'):
        if target.is_file():
            if sha256(target) == checksum or canonical_json_sha256(json.loads(target.read_text())) == R2R_MIRROR[split][1]:
                return target
            raise ValueError(f'R2R annotation checksum mismatch: {target}')
        try:
            return fetch(f'https://dl.dropboxusercontent.com/s/{token}/R2R_{split}.json', target, checksum, download)
        except requests.RequestException:
            if not download:
                raise
            url = f'https://huggingface.co/datasets/thomas-yanxin/r2r-metadata/resolve/{R2R_MIRROR_REVISION}/{split}.jsonl'
            mirror = fetch(url, root/f'{split}.jsonl', R2R_MIRROR[split][0])
            records = [json.loads(line) for line in mirror.read_text().splitlines()]
            if canonical_json_sha256(records) != R2R_MIRROR[split][1]:
                raise ValueError('R2R mirror differs from the official records')
            temporary = target.with_suffix('.partial')
            temporary.write_text(json.dumps(records, ensure_ascii=True)+'\n')
            temporary.replace(target)
            (root/f'{split}-source.json').write_text(json.dumps({'mirror': url, 'mirror_sha256': R2R_MIRROR[split][0],
                'official_sha256': checksum, 'canonical_sha256': R2R_MIRROR[split][1]}, indent=2)+'\n')
            return target


def vlnce_annotations(split='train', root=None, download=True):
    from filelock import FileLock
    if split not in ('train', 'val_seen', 'val_unseen'):
        raise ValueError('Unsupported VLN-CE split')
    root = Path(root) if root else dataset_root()/'R2R_VLNCE_v1-3_preprocessed'
    annotation = root/split/f'{split}.json.gz'
    if annotation.is_file() and (split != 'train' or (root/'train/train_gt.json.gz').is_file()):
        return annotation
    archive = fetch('https://drive.usercontent.google.com/download?id=1fo8F4NKgZDH-bPSdVU3cONAkt5EW-tyr&export=download&confirm=t',
                    cache_root()/'downloads/R2R_VLNCE_v1-3_preprocessed.zip', VLNCE_SHA, download)
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root)+'.lock'), zipfile.ZipFile(archive) as source:
        if annotation.is_file() and (split != 'train' or (root/'train/train_gt.json.gz').is_file()):
            return annotation
        # Copy data only, rejecting paths outside the release root.
        for name in source.namelist():
            parts = Path(name).parts
            if 'R2R_VLNCE_v1-3_preprocessed' in parts:
                parts = parts[parts.index('R2R_VLNCE_v1-3_preprocessed')+1:]
            if not parts or name.endswith('/'):
                continue
            target = root.joinpath(*parts).resolve()
            if not target.is_relative_to(root.resolve()):
                raise ValueError('Unsafe annotation archive path')
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name+'.partial')
            with source.open(name) as incoming, temporary.open('wb') as output:
                import shutil
                shutil.copyfileobj(incoming, output)
            temporary.replace(target)
    if not annotation.is_file():
        raise FileNotFoundError(f'Official archive lacks {annotation}')
    return annotation


def panorama_features(path=None, download=True):
    from filelock import FileLock
    root = Path(path) if path else dataset_root()/'places365'
    if (root/'index.json').is_file() and (root/'features.npy').is_file():
        return root
    archive = fetch('https://dl.dropboxusercontent.com/s/85tpa6tc3enl5ud/ResNet-152-places365.zip',
                    cache_root()/'downloads/ResNet-152-places365.zip', FEATURE_SHA, download)
    root.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root)+'.lock'):
        from vlastudio.data_utils.datasets.navigation_features import convert_features
        convert_features(archive, root, FEATURE_SHA)
    return root


class _RemoteZip(io.RawIOBase):
    """Read the published ZIP member without downloading the unused checkpoint."""
    def __init__(self, url):
        import requests
        self.url, self.pos = url, 0
        response = requests.get(url, headers={'Range': 'bytes=0-0'}, timeout=(20, 90))
        response.raise_for_status()
        if response.status_code != 206:
            raise RuntimeError('Official checkpoint server must support HTTP Range')
        self.size = int(response.headers['Content-Range'].split('/')[-1])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos+offset if whence == 1 else self.size+offset
        return self.pos

    def read(self, size=-1):
        import requests
        size = min(size if size >= 0 else self.size-self.pos, self.size-self.pos)
        if size <= 0:
            return b''
        response = requests.get(self.url, headers={'Range': f'bytes={self.pos}-{self.pos+size-1}'}, timeout=(20, 180))
        response.raise_for_status()
        if response.status_code != 206 or len(response.content) != size:
            raise RuntimeError('Invalid checkpoint archive range')
        self.pos += size
        return response.content


def official_checkpoint(name, download=True):
    from filelock import FileLock
    root = cache_root()/'models'/name
    root.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root)+'.lock'):
        required = ['policy.pt', 'config.json', 'policy_metadata.json', 'normalize.json']
        if name == 'recurrent_vlnbert':
            required.append('vocab.txt')
        if all((root/file).is_file() for file in required):
            return root
        if name == 'vlnce_cma':
            source = fetch('https://drive.usercontent.google.com/download?id=1o9PgBT38BH9pw_7V1QB3XUkY8auJqGKw&export=download&confirm=t',
                           cache_root()/'downloads/cma-pm-da-aug.pth', CMA_SHA, download)
            from vlastudio.policy.vlnce_cma.pretrained import convert_checkpoint
            convert_checkpoint(source, root, CMA_SHA)
        elif name == 'recurrent_vlnbert':
            source = cache_root()/'downloads/vlnbert-prevalent.pt'
            if not source.is_file():
                if not download:
                    raise FileNotFoundError(f'Checkpoint missing with download=false: {source}')
                source.parent.mkdir(parents=True, exist_ok=True)
                reader = _RemoteZip('https://zenodo.org/records/4437864/files/snap.zip?download=1')
                temporary = source.with_suffix('.partial')
                with zipfile.ZipFile(reader) as archive, archive.open('snap/VLNBERT-PREVALENT-final/state_dict/best_val_unseen') as incoming, temporary.open('wb') as output:
                    import shutil
                    shutil.copyfileobj(incoming, output, 8*1024*1024)
                if sha256(temporary) != BERT_SHA:
                    temporary.unlink()
                    raise ValueError('Official VLN-BERT checkpoint checksum mismatch')
                temporary.replace(source)
            vocabulary = fetch('https://huggingface.co/google-bert/bert-base-uncased/resolve/main/vocab.txt',
                               cache_root()/'downloads/vocab.txt', VOCAB_SHA, download)
            from vlastudio.policy.recurrent_vlnbert.pretrained import convert_checkpoint
            convert_checkpoint(source, root, BERT_SHA, vocabulary)
        else:
            raise ValueError(f'Unknown navigation model: {name}')
    return root


def resolve_checkpoint(reference, name, download=True):
    if reference == 'official':
        return official_checkpoint(name, download)
    path = Path(reference).expanduser()
    if path.exists():
        return path
    # A configured HF repository is also a valid pretrained-weight source.
    if '/' in str(reference) and not path.is_absolute() and not str(reference).startswith(('.', '~')):
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(str(reference), local_files_only=not download))
    raise FileNotFoundError(f'Navigation checkpoint not found: {reference}')
