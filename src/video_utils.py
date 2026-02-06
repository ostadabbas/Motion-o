from typing import List, Tuple, Optional, Any
from PIL import Image
import numpy as np
try:
    # MoviePy v2+ API
    from moviepy import VideoFileClip  # type: ignore
except Exception:
    # Fallback for MoviePy v1.x
    from moviepy.editor import VideoFileClip  # type: ignore


def extract_frames_between(video_path: str,
                           start_time: float,
                           end_time: float,
                           fps: float = 0.25,
                           max_frames: int = 4,
                           max_size: int = 512,
                           clip: Optional[Any] = None) -> List[Tuple[Image.Image, float]]:
    """
    Extract frames between start_time and end_time at a given sampling fps.
    Returns list of tuples (PIL.Image, timestamp_seconds).
    
    Args:
        video_path: Path to video file (ignored if clip is provided)
        start_time: Start timestamp in seconds
        end_time: End timestamp in seconds
        fps: Sampling rate (frames per second)
        max_frames: Maximum number of frames to extract
        max_size: Maximum size for resizing (maintains aspect ratio)
        clip: Optional VideoFileClip object to reuse (avoids reopening video)
    """
    assert end_time > start_time, "end_time must be greater than start_time"
    
    # Use provided clip or create new one
    should_close = False
    if clip is None:
        clip = VideoFileClip(video_path)
        should_close = True
    
    duration = clip.duration
    s = max(0.0, float(start_time))
    e = min(float(end_time), duration)
    if e <= s:
        if should_close:
            clip.close()
        return []

    step = 1.0 / max(fps, 1e-6)
    t = s
    frames: List[Tuple[Image.Image, float]] = []
    while t <= e:
        frame = clip.get_frame(t)  # RGB numpy array
        img = Image.fromarray(frame.astype(np.uint8))
        # Resize to keep shorter side <= max_size (maintain aspect)
        w, h = img.size
        scale = max(1.0, min(w, h) / float(max_size))
        if scale > 1.0:
            new_w = int(round(w / scale))
            new_h = int(round(h / scale))
            img = img.resize((new_w, new_h), Image.BILINEAR)
        frames.append((img, float(t)))
        if len(frames) >= max_frames:
            break
        t += step

    if should_close:
        clip.close()
    return frames


def build_collages(frames: List[Tuple[Image.Image, float]],
                   grid_cols: int = 2,
                   grid_rows: int = 2,
                   max_collages: int = 3,
                   tile_size: int = 256) -> List[Tuple[Image.Image, float]]:
    """
    Pack many frames into a few tiled collage images to keep VRAM use stable.
    Returns list of (collage_image, timestamp_of_last_tile_in_collage).
    """
    if not frames:
        return []
    tiles_per_collage = max(1, grid_cols * grid_rows)
    result: List[Tuple[Image.Image, float]] = []
    idx = 0
    for _ in range(max_collages):
        if idx >= len(frames):
            break
        canvas = Image.new("RGB", (grid_cols * tile_size, grid_rows * tile_size), (0, 0, 0))
        last_ts = frames[min(len(frames)-1, idx + tiles_per_collage - 1)][1]
        for r in range(grid_rows):
            for c in range(grid_cols):
                if idx >= len(frames):
                    break
                img, ts = frames[idx]
                # Fit into tile
                w, h = img.size
                scale = max(w / tile_size, h / tile_size, 1.0)
                nw, nh = int(round(w / scale)), int(round(h / scale))
                thumb = img.resize((nw, nh), Image.BILINEAR)
                x = c * tile_size + (tile_size - nw) // 2
                y = r * tile_size + (tile_size - nh) // 2
                canvas.paste(thumb, (x, y))
                idx += 1
        result.append((canvas, float(last_ts)))
    return result


def sample_full_video_as_collages(video_path: str,
                                  pause_start: float,
                                  fps: float = 0.2,
                                  max_frames: int = 40,
                                  grid_cols: int = 2,
                                  grid_rows: int = 2,
                                  max_collages: int = 3,
                                  tile_size: int = 256,
                                  clip: Optional[Any] = None) -> List[Tuple[Image.Image, float]]:
    """
    Sample frames from [0, pause_start] then pack them into a few collages.
    
    Args:
        clip: Optional VideoFileClip object to reuse (avoids reopening video)
    """
    if pause_start <= 0:
        return []
    raw = extract_frames_between(video_path, 0.0, pause_start, fps=fps, max_frames=max_frames, max_size=tile_size, clip=clip)
    collages = build_collages(raw, grid_cols=grid_cols, grid_rows=grid_rows, max_collages=max_collages, tile_size=tile_size)
    return collages
