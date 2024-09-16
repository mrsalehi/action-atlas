"""
Script to extract the segments in ActionAtlas that public will evaluate on.
The segments are extracted from the downloaded YouTube videos.
"""
from pathlib import Path
from typing import List
import os
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from loguru import logger
from tqdm import tqdm

from action_atlas.utils import (
    extract_video_segment, 
    get_video_duration, 
    resolve_media_path,
    read_jsonl
)


def extract_action_atlas_final_segments_multithread(
    data_fpath: str,
    yt_videos_dir: str,
    out_segments_dir: str,
    max_workers: int=32,
    overwrite: bool=True,
) -> int:
    """
    Extract segments for ActionAtlas evaluation.

    Args:
        data_fpath (str): Path to the JSONL file containing the final ActionAtlas data.
        yt_videos_dir (str): Directory containing YouTube videos.
        out_segments_dir (str): Directory to save the extracted segments.
        max_workers (int): Number of workers for multithreading.
        overwrite (bool): Whether to overwrite existing segments.

    Returns:
        int: 0 if successful, -1 if there were errors.
    """
    def _extract_segment(video_fpath: str, start: float, end: float) -> None:
        yt_id = Path(video_fpath).stem
        segment_name = f"{yt_id}_{start}_{end}.mp4"
        segment_path = Path(out_segments_dir) / segment_name

        if segment_path.exists() and not overwrite:
            logger.info(f"Segment {segment_name} already exists.")

        video_fpath = resolve_media_path(video_fpath)

        if video_fpath == -1:
            return -1

        video_duration = get_video_duration(video_fpath)
        end = min(end, float(video_duration))

        try:
            ret = extract_video_segment(
                src_path=video_fpath,
                des_path=segment_path.as_posix(),
                start=start,
                duration=end - start,
                quiet=True,
                extract_audio=False,
            )
            return ret
        except Exception as e:
            logger.error(f"Error in extracting segment {yt_id}_{start}_{end}: {e}")
            return -1
        
    data = read_jsonl(data_fpath)

    videos = [video for video in Path(yt_videos_dir).glob("*") \
        if video.suffix in {".mp4", ".mkv", ".webm"}]
    
    yt_id_to_local_video_path = {v_path.stem: v_path.as_posix() for v_path in videos}
    for d in data:
        d.update({"video_path": yt_id_to_local_video_path[d["youtube_id"]]})

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _extract_segment,
                el["video_path"],
                round(float(el["start_timestamp"]), 1),
                round(float(el["end_timestamp"]), 1),
            )
            for el in data
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting segments"):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error in future: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_fpath", type=str, required=True, help="Path to the jsonl file released by ActionAtlas team containing the metadata.")
    parser.add_argument("--yt_videos_dir", type=str, required=True, help="Path to the directory containing the downloaded YouTube videos.")
    parser.add_argument("--out_segments_dir", type=str, required=True, help="Path to the directory to save the extracted segments in ActionAtlas.")
    parser.add_argument("--max_workers", type=int, default=32, help="Number of workers to use for multithreading.")

    args = parser.parse_args()
    extract_action_atlas_final_segments_multithread(
        data_fpath=args.data_fpath,
        yt_videos_dir=args.yt_videos_dir,
        out_segments_dir=args.out_segments_dir,
        max_workers=args.max_workers,
        overwrite=False
    )