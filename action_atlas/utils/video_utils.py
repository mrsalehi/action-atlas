import subprocess
import os
from pathlib import Path
from typing import Union
from concurrent.futures import ThreadPoolExecutor
import base64
from PIL import Image
from io import BytesIO
import uuid
from typing import Optional, Union

import cv2
import numpy as np
import wget
from tqdm import tqdm
from loguru import logger

from action_atlas.utils import (
    download_gcs_blob,
    read_jsonl,
    upload_gcs_blob
)


def resolve_media_path(
    media_path: Optional[str] = None, 
    image_str: Optional[str] = None, 
    image_ext: Optional[str] = None, 
    verbose: bool = True) -> Union[str, int]:
    """
    If media_path is a gcs path, then download the media (video/image) into a temp file.
    If the media_path is a local path, then use the path as is.
    In any case, if the media does not exist, return -1.
    You could also provide image_str which is the image representation in 
    encoded string format. In that case, the image will be decoded and saved
    in a tempfile. Also you can't provide both image_str and media_path.
    I had to add the image_str option because the M3IT data saves images
    as base64 strings in the jsonl files.
    """
    if image_str:
        if media_path:
            raise ValueError("Cannot provide both image_str and media_path.")
        if not image_ext or not image_ext.startswith("."):
            raise ValueError("Invalid or missing image extension.")
        
        try:
            image_data = base64.b64decode(image_str)
            image = Image.open(BytesIO(image_data)).convert("RGB")
            file_path = Path('/tmp') / f"{uuid.uuid4()}{image_ext}"
            image.save(file_path)
            return str(file_path)
        except Exception as e:
            logger.error(f"Failed to decode and save image: {e}")
            return -1

    out_path = Path("/tmp") / media_path.split("/")[-1]
    if media_path.startswith('gs'):
        ret = download_gcs_blob(
            blob_path=media_path,
            destination_file_name=str(out_path),
            verbose=False
        )
        if ret != 0:
            logger.error(f"Failed to download {media_path}.")
            return -1
    elif media_path.startswith("http"):
        try:
            wget.download(media_path, out=str(out_path), bar=lambda current, total, width: None)
        except Exception as e:
            if verbose:
                logger.error(f"Could not download {media_path} to {out_path}: {e}")
            return -1
    else:
        if not Path(media_path).exists():
            if verbose:
                logger.error(f"{media_path} does not exist.")
            return -1
        return media_path

    return str(out_path)


def get_frame_rate(video_path: str) -> float:
    """Get the frame rate of a video file using OpenCV"""
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return frame_rate


def frames_to_time(frame_number: int, frame_rate: float) -> float:
    seconds = frame_number / frame_rate
    return seconds


def is_valid_video(file_path: str) -> bool:
    """Validates the video file using ffmpeg."""
    try:
        subprocess.run(
            ['ffmpeg', '-v', 'error', '-i', file_path, '-f', 'null', '-'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )
        return True
    except subprocess.CalledProcessError:
        return False
    except Exception as e:
        logger.error(f"Error validating video {file_path}: {e}")
        return False


def extract_video_segment(
    src_path: str,
    des_path: str,
    start: float,
    duration: float,
    quiet: bool=False,
    max_retries: int=5,
    extract_audio: bool=True,
    seek_to_keyframes: bool=False,
    ):
    """ffmpeg wrapper to extract a video segment from a video file."""
    base_cmd = f"ffmpeg -y {'-ss' if seek_to_keyframes else '-accurate_seek -i'} {src_path} -ss {start} -t {duration} -c:v libx264"
    base_cmd += " -c:a aac -b:a 96k" if extract_audio else " -an"
    base_cmd += " -loglevel quiet" if quiet else ""
    base_cmd += f" {des_path}"

    for attempt in range(max_retries):
        # check if the extracted segment is valid using ffprobe
        try:
            subprocess.run(base_cmd, shell=True, check=True)

            command = [
                'ffmpeg',
                '-v', 'error',
                '-i', des_path,
                '-f', 'null',
                '-'
            ]
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            # if exits without an error, the file is considered valid
            if result.returncode == 0:
                return 0
        except subprocess.CalledProcessError as e:
            if attempt == max_retries - 1:
                logger.error(f"Failed to extract the video segment from {src_path} to {des_path} after {max_retries} attempts.")
                raise e

    # logger.error(f"Failed to extract the video segment from {src_path} to {des_path}.")
    # return -1


def get_video_duration(file_path: str) -> float:
    """Get the duration of a video file using ffprobe."""
    command = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        file_path
    ]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return float(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        logger.error(f"Error getting video duration for {file_path}: {e}")
        return 0.0

    
def extract_video_segment_by_frames_vfr(src_path, des_path, start_frame, end_frame, quiet=False, max_retries=5, extract_audio=True):
    """Extract a video segment with frame numbers from a video with variable 
    frame rate (VFR) using ffmpeg"""
    base_cmd = f"ffmpeg -y -accurate_seek -i {src_path} -vf 'select=between(n\\,{start_frame}\\,{end_frame}),setpts=PTS-STARTPTS'"
    if extract_audio:
        base_cmd += " -af 'aselect=between(n\\,{start_frame}\\,{end_frame}),asetpts=PTS-STARTPTS' -c:v libx264 -c:a aac -b:a 96k"
    else:
        base_cmd += " -an -c:v libx264" 
    if quiet:
        base_cmd += " -loglevel quiet"
    base_cmd += f" {des_path}"

    retry = 0
    for retry in range(max_retries):
        # check if the extracted segment is valid using ffprobe
        try:
            command = [
                'ffmpeg',
                '-v', 'error',  
                '-i', des_path,  
                '-f', 'null',  # output to null format 
                '-'  # direct output to nowhere (stdout)
            ]
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            return 0
        except subprocess.CalledProcessError as e:
            logger.warning(f"Error occurred when extracting video segment from {src_path} to {des_path}: {e}. Retrying...")
    
    logger.error(f"Failed to extract the video segment from {src_path} to {des_path}.")
    return -1


def reencode_video(input_video_fpath: str, output_video_fpath: str) -> None:
    """
    Ensures that the video codec is compatible for browser playback using ffmpeg.
    
    This function re-encodes the video to use the H.264 codec for video and AAC codec for audio,
    which are widely supported by browsers. Note that ffmpeg cannot process the video in-place,
    so the output video must be written to a separate file.

    Args:
        input_video_fpath (str): Path to the input video file.
        output_video_fpath (str): Path to the output video file.
    """
    input_path = Path(input_video_fpath)
    output_path = Path(output_video_fpath)

    # Ensure the parent directory of the output file exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        'ffmpeg', '-loglevel', 'quiet', '-y', '-i', input_path.as_posix(),
        '-c:v', 'libx264', '-preset', 'slow', '-crf', '22',
        '-c:a', 'aac', '-strict', '-2', output_path.as_posix()
    ]

    # Run the ffmpeg command
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while processing video: {e}")
        raise


def convert_to_mp4(video_fpath: str, mp4_fpath: str, max_retries: int = 5) -> int:
    assert Path(video_fpath).suffix in {".avi", ".mkv", ".flv", ".mov", ".wmv", ".mpg", ".mpeg", ".mp4"}
    assert Path(mp4_fpath).suffix == ".mp4"
    
    if video_fpath == mp4_fpath:
        return 0
     
    cmd = ['ffmpeg', '-i', video_fpath, '-y', '-c:v', 'libx264', '-c:a', 'aac', '-b:a', '96k', mp4_fpath]

    for _ in range(max_retries):
        # check if the extracted segment is valid using ffprobe
        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            return 0
        except subprocess.CalledProcessError as e:
            logger.warning(f"Error occurred when converting {video_fpath} to mp4: {e}. Retrying...")
            
    return  -1


def convert_videos_to_one_fps_multithread(
    benchmark_jsonl_fpath: Union[str, Path],
    fix_codec: bool=True
    ):
    """Convert videos in the benchmark jsonl file to one fps using cv2.
    This is used for experiments with Gemini and any video model exposed just
    as an API.
    """
    # Capture the video from the input file
    def _convert_video_to_one_fps(video_path: str):
        video_fname = Path(video_path).name
        input_video_path = f"/tmp/one_fps/{video_fname}"
        os.makedirs("/tmp/one_fps", exist_ok=True)
        output_video_path = input_video_path.replace('.mp4', '_one_fps.mp4')
        if fix_codec:
            output_video_path = input_video_path.replace('.mp4', '_bad_codec_one_fps.mp4')

        video_path = resolve_media_path(video_path)
        cap = cv2.VideoCapture(input_video_path)
 
        if not cap.isOpened():
            print("Error: Could not open video.")
            return

        # Get the width and height of the video
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change 'mp4v' to 'XVID' if you prefer
        out = cv2.VideoWriter(output_video_path, fourcc, 1, (frame_width, frame_height))

        while True:
            # Read the frame
            ret, frame = cap.read()
            if not ret:
                break 
            out.write(frame)

        cap.release()
        out.release()

        if fix_codec:
            reencode_video(
                input_video_fpath=output_video_path,
                output_video_fpath=input_video_path.replace('.mp4', '_one_fps.mp4'),
            )
            output_video_path = input_video_path.replace('.mp4', '_one_fps.mp4')
 
        upload_gcs_blob(
            bucket_name="video_llm",
            source_file_name=output_video_path,
            destination_blob_name=f"data/action-benchmark/final_segments_no_audio_blurred_one_fps/{Path(output_video_path).name}",
        )
 
    data = read_jsonl(benchmark_jsonl_fpath)
    paths = [d["video_path"] for d in data]
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = []
        for path in paths:
            futures.append(executor.submit(_convert_video_to_one_fps, path))
        for future in tqdm(futures, desc="Converting videos to one fps"):
            future.result()


def create_horizontal_storyboard(video_path, output_image_path, num_frames=9):
    """Simple util to create a storyboard figure for papers by sampling frames"""
    # Load the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get the total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the frame indices to sample
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    # List to store the sampled frames
    frames = []

    for idx in frame_indices:
        # Set the video position to the frame index
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Could not read frame at index {idx}.")
            continue
        frames.append(frame)

    # Release the video capture object
    cap.release()

    # Concatenate frames horizontally
    storyboard = np.hstack(frames)

    # Save the resulting image
    cv2.imwrite(output_image_path, storyboard)
    print(f"Storyboard saved to {output_image_path}")
