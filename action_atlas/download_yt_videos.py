import argparse
import subprocess
import json
from pathlib import Path
from loguru import logger
from typing import List


def download_video(yt_id: str, out_dir: str) -> None:
    """Download a single video from YouTube using yt-dlp."""
    url = f"https://www.youtube.com/watch?v={yt_id}"
    cmd = [
        'yt-dlp',
        '--format',
        'bestvideo[height<=720]+bestaudio/best[height<=720]/best',
        '-o', f"{out_dir}/{yt_id}.%(ext)s",
        url
    ]

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"Successfully downloaded video {yt_id}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error in downloading video {yt_id}: {e.stderr}")


def read_yt_ids_from_jsonl(jsonl_fpath: str) -> List[str]:
    """Read YouTube video IDs from a JSONL file."""
    yt_ids = []
    with open(jsonl_fpath, 'r') as file:
        for line in file:
            data = json.loads(line)
            yt_ids.append(data['youtube_id'])
    return yt_ids


def main(yt_ids_jsonl_fpath: str, out_dir: str) -> None:
    """Download videos from YouTube using yt-dlp."""
    yt_ids = read_yt_ids_from_jsonl(yt_ids_jsonl_fpath)
    for yt_id in yt_ids:
        download_video(yt_id, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download YouTube videos using yt-dlp.")
    parser.add_argument("--yt_ids_jsonl_fpath", type=str, required=True, help="Path to the JSONL file containing the YouTube video IDs.")
    parser.add_argument("--out_dir", type=str, default="/tmp", help="Directory to save the downloaded videos.")
    args = parser.parse_args()

    main(args.yt_ids_jsonl_fpath, args.out_dir)