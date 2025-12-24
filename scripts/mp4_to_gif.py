#!/usr/bin/env python3
"""
Convert MP4 video files to GIF format.

This script converts MP4 video files to animated GIF format with customizable
parameters such as resolution, frame rate, and duration.

Usage examples:
    # Convert a single MP4 file to GIF
    python mp4_to_gif.py input.mp4

    # Convert with custom output filename
    python mp4_to_gif.py input.mp4 -o output.gif

    # Convert with reduced resolution and frame rate
    python mp4_to_gif.py input.mp4 --fps 10 --scale 0.5

    # Convert only first 5 seconds
    python mp4_to_gif.py input.mp4 --duration 5

    # Convert multiple files
    python mp4_to_gif.py video1.mp4 video2.mp4 video3.mp4
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

try:
    from moviepy.editor import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    try:
        import imageio
        import imageio_ffmpeg
        IMAGEIO_AVAILABLE = True
    except ImportError:
        IMAGEIO_AVAILABLE = False


def convert_mp4_to_gif_moviepy(
    input_path: Path,
    output_path: Path,
    fps: int = 10,
    scale: float = 1.0,
    duration: Optional[float] = None,
    start_time: float = 0.0,
) -> None:
    """
    Convert MP4 to GIF using moviepy.
    
    Args:
        input_path: Path to input MP4 file
        output_path: Path to output GIF file
        fps: Frames per second for output GIF
        scale: Scale factor for resolution (1.0 = original, 0.5 = half size)
        duration: Maximum duration in seconds (None = full video)
        start_time: Start time in seconds
    """
    print(f"Loading video: {input_path}")
    clip = VideoFileClip(str(input_path))
    
    # Apply start time and duration
    if start_time > 0:
        clip = clip.subclip(start_time)
    if duration is not None:
        clip = clip.subclip(0, duration)
    
    # Apply scaling
    if scale != 1.0:
        clip = clip.resize(scale)
    
    # Set FPS
    clip = clip.set_fps(fps)
    
    print(f"Converting to GIF: {output_path}")
    print(f"  Resolution: {clip.size}")
    print(f"  FPS: {fps}")
    print(f"  Duration: {clip.duration:.2f}s")
    
    clip.write_gif(
        str(output_path),
        fps=fps,
        program='ffmpeg',
        opt='optimizeplus',
        verbose=False,
        logger=None
    )
    
    clip.close()
    print(f"✓ Successfully created: {output_path}")


def convert_mp4_to_gif_imageio(
    input_path: Path,
    output_path: Path,
    fps: int = 10,
    scale: float = 1.0,
    duration: Optional[float] = None,
    start_time: float = 0.0,
) -> None:
    """
    Convert MP4 to GIF using imageio (fallback method).
    
    Args:
        input_path: Path to input MP4 file
        output_path: Path to output GIF file
        fps: Frames per second for output GIF
        scale: Scale factor for resolution (1.0 = original, 0.5 = half size)
        duration: Maximum duration in seconds (None = full video)
        start_time: Start time in seconds
    """
    import imageio
    import numpy as np
    from PIL import Image
    
    print(f"Loading video: {input_path}")
    reader = imageio.get_reader(str(input_path), fps=fps)
    
    # Get video metadata
    meta = reader.get_meta_data()
    original_fps = meta.get('fps', fps)
    
    # Try to get total frames (may not be available for all formats)
    try:
        total_frames = reader.count_frames()
    except (AttributeError, TypeError):
        # For formats that don't support count_frames(), we'll read until we can't
        total_frames = None
    
    # Calculate frame range
    start_frame = int(start_time * original_fps)
    if duration is not None:
        if total_frames is not None:
            end_frame = min(int((start_time + duration) * original_fps), total_frames)
        else:
            end_frame = int((start_time + duration) * original_fps)
    else:
        end_frame = total_frames
    
    # Calculate frame step to match target FPS
    frame_step = max(1, int(original_fps / fps))
    
    frames = []
    try:
        for i, frame in enumerate(reader):
            if i < start_frame:
                continue
            if end_frame is not None and i >= end_frame:
                break
            if (i - start_frame) % frame_step == 0:
                # Apply scaling
                if scale != 1.0:
                    img = Image.fromarray(frame)
                    new_size = (int(img.width * scale), int(img.height * scale))
                    frame = np.array(img.resize(new_size, Image.Resampling.LANCZOS))
                frames.append(frame)
    except (EOFError, StopIteration):
        # End of video reached
        pass
    
    reader.close()
    
    if not frames:
        raise ValueError("No frames extracted from video")
    
    print(f"Converting to GIF: {output_path}")
    print(f"  Resolution: {frames[0].shape[1]}x{frames[0].shape[0]}")
    print(f"  FPS: {fps}")
    print(f"  Frames: {len(frames)}")
    
    imageio.mimsave(
        str(output_path),
        frames,
        fps=fps,
        loop=0,  # Infinite loop
    )
    
    print(f"✓ Successfully created: {output_path}")


def convert_mp4_to_gif(
    input_path: Path,
    output_path: Optional[Path] = None,
    fps: int = 10,
    scale: float = 1.0,
    duration: Optional[float] = None,
    start_time: float = 0.0,
) -> None:
    """
    Convert MP4 to GIF using available library.
    
    Args:
        input_path: Path to input MP4 file
        output_path: Path to output GIF file (None = auto-generate)
        fps: Frames per second for output GIF
        scale: Scale factor for resolution
        duration: Maximum duration in seconds
        start_time: Start time in seconds
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Check file extension (basic validation)
    valid_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv', '.m4v'}
    file_ext = input_path.suffix.lower()
    
    # Skip GIF files (already in GIF format)
    if file_ext == '.gif':
        raise ValueError(
            f"Input file is already a GIF: {input_path}. "
            f"Skipping conversion (GIF to GIF conversion is not supported)."
        )
    
    if file_ext not in valid_extensions:
        raise ValueError(
            f"Unsupported file format: {file_ext}. "
            f"Supported formats: {', '.join(sorted(valid_extensions))}"
        )
    
    if output_path is None:
        output_path = input_path.with_suffix('.gif')
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if MOVIEPY_AVAILABLE:
        convert_mp4_to_gif_moviepy(input_path, output_path, fps, scale, duration, start_time)
    elif IMAGEIO_AVAILABLE:
        convert_mp4_to_gif_imageio(input_path, output_path, fps, scale, duration, start_time)
    else:
        raise ImportError(
            "Neither moviepy nor imageio is available. "
            "Please install one of them:\n"
            "  pip install moviepy\n"
            "  or\n"
            "  pip install imageio imageio-ffmpeg"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Convert MP4 video files to GIF format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single file
  python mp4_to_gif.py video.mp4

  # Convert with custom output
  python mp4_to_gif.py video.mp4 -o output.gif

  # Convert with reduced quality (faster, smaller file)
  python mp4_to_gif.py video.mp4 --fps 10 --scale 0.5

  # Convert first 5 seconds only
  python mp4_to_gif.py video.mp4 --duration 5

  # Convert multiple files
  python mp4_to_gif.py video1.mp4 video2.mp4 video3.mp4
        """
    )
    
    parser.add_argument(
        'input_files',
        nargs='+',
        type=Path,
        help='Input MP4 file(s) to convert'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=Path,
        help='Output GIF file path (default: same as input with .gif extension)'
    )
    
    parser.add_argument(
        '--fps',
        type=int,
        default=10,
        help='Frames per second for output GIF (default: 10)'
    )
    
    parser.add_argument(
        '--scale',
        type=float,
        default=1.0,
        help='Scale factor for resolution (1.0 = original, 0.5 = half size) (default: 1.0)'
    )
    
    parser.add_argument(
        '--duration',
        type=float,
        default=None,
        help='Maximum duration in seconds (default: full video)'
    )
    
    parser.add_argument(
        '--start-time',
        type=float,
        default=0.0,
        help='Start time in seconds (default: 0.0)'
    )
    
    args = parser.parse_args()
    
    # Check if required libraries are available
    if not MOVIEPY_AVAILABLE and not IMAGEIO_AVAILABLE:
        print("ERROR: Neither moviepy nor imageio is available.", file=sys.stderr)
        print("Please install one of them:", file=sys.stderr)
        print("  pip install moviepy", file=sys.stderr)
        print("  or", file=sys.stderr)
        print("  pip install imageio imageio-ffmpeg", file=sys.stderr)
        sys.exit(1)
    
    # Process each input file
    for i, input_file in enumerate(args.input_files):
        if len(args.input_files) > 1:
            print(f"\n[{i+1}/{len(args.input_files)}] Processing: {input_file}")
        
        # Determine output path
        if args.output:
            if len(args.input_files) > 1:
                # Multiple files: append index to output name
                output_file = args.output.parent / f"{args.output.stem}_{i+1}{args.output.suffix}"
            else:
                output_file = args.output
        else:
            output_file = None
        
        try:
            convert_mp4_to_gif(
                input_path=input_file,
                output_path=output_file,
                fps=args.fps,
                scale=args.scale,
                duration=args.duration,
                start_time=args.start_time,
            )
        except ValueError as e:
            # For unsupported formats (like GIF), print warning and continue
            error_msg = str(e)
            if 'already a GIF' in error_msg or 'Unsupported file format' in error_msg:
                print(f"SKIP: {error_msg}", file=sys.stderr)
            else:
                print(f"ERROR: Failed to convert {input_file}: {e}", file=sys.stderr)
            if len(args.input_files) > 1:
                continue
            else:
                sys.exit(1)
        except Exception as e:
            print(f"ERROR: Failed to convert {input_file}: {e}", file=sys.stderr)
            if len(args.input_files) > 1:
                print("Continuing with next file...", file=sys.stderr)
                continue
            else:
                sys.exit(1)


if __name__ == '__main__':
    main()

