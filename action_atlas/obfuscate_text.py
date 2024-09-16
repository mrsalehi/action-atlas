"""
This script obfuscates text in videos that contains words. To do so we followed this procedure:
1. identify videos with text matching subwords of their ground truth action names
2. identify videos flagged by human annotators as needing text filtering
3. Using Google Cloud Vision API and OpenCV, we obfuscate detected text in all frames of both types of videos 
if it shares words with any collected action name.

We have provided the masks in the released data. Function `obfuscate_text_in_videos_with_masks_multithread`
uses already detected mask coordinates to obfuscate text.
"""
import os
import json
import io
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from typing import Union, List, Dict
import shutil

from tqdm import tqdm
from google.cloud import vision
from loguru import logger
from google.cloud.vision import AnnotateImageResponse, EntityAnnotation
import cv2

from action_atlas.utils import (
    reencode_video,
    resolve_media_path,
    read_jsonl
)


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "<PATH_TO_YOUR_CREDENTIALS>"


def detect_text(image: cv2.Mat) -> List[EntityAnnotation]:
    """Detects text in the image using Google Cloud Vision API."""
    client = vision.ImageAnnotatorClient()

    success, encoded_image = cv2.imencode('.png', image)
    if not success:
        raise ValueError("Error in encoding the image.")

    content = encoded_image.tobytes()
    image = vision.Image(content=content)

    response: AnnotateImageResponse = client.text_detection(image=image)
    if response.error.message:
        raise RuntimeError(f"Error in detecting text: {response.error.message}")

    texts = response.text_annotations
    texts = texts[1:] if len(texts) > 1 and len(texts[0].description) > 1 else texts

    if response.error.message:
        raise Exception(f'{response.error.message}')

    return texts


def obfuscate_text_areas(frame, texts):
    """Obfuscates areas of the frame that has text with letters."""
    for text in texts:
        if any(c.isalpha() for c in text["description"]):
            vertices = [xy for xy in text["bounding_poly"]["vertices"]]
            x_min = min(vertices, key=lambda v: v[0])[0]
            y_min = min(vertices, key=lambda v: v[1])[1]
            x_max = max(vertices, key=lambda v: v[0])[0]
            y_max = max(vertices, key=lambda v: v[1])[1]

            roi = frame[y_min:y_max, x_min:x_max]
            blurred_roi = cv2.GaussianBlur(roi, (111, 111), 0)
            frame[y_min:y_max, x_min:x_max] = blurred_roi

    return frame


def obfuscate_text_in_videos_gcloud_vision_api(
    jsonl_fpath: str,
    max_workers: int,
    out_dir: str,
    video_segments_dir: str,
    debug: bool = False
    ):
    os.makedirs(out_dir, exist_ok=True)
    all_action_names = json.load(open("../data/all_actions.json"))
    all_action_names = [v["action"] for v in all_action_names.values()]
    all_action_names_subwords = set([word.lower() for action in all_action_names for word in action.split()])
    all_action_names_subwords.update({
        "legal", "north", "south", "explosive", "not", "legal",
        "backheel", "!", "trickshot", "faint", "Ganda", "ng",
        "ni", "Idol", "carry!", "fancy", "skills",
    })

    def _blur_text_single_video(
        metadata: Dict
    ):
        """Does the following:
        1. Selects videos whose frames should be obfuscated:
            1.1. Checks every 10 frames and if any of those frames has text that contains words from
            the ground truth action name the video is selected.
            1.2. If the video was flaged by annotator with the comment "filter text" in the jsonl file, 
            then it's also selected.
        2. Iterates over all frames of selected videos and obfuscates any text that has overlap with any subword 
            of all action names.
        """
        try:
            video_path = resolve_media_path(metadata["video_path"])
        except Exception as e:
            logger.error(f"Error in getting video path: {e}")
            return -1

        video_stem = Path(video_path).stem

        blur_metadata_out_path = os.path.join(out_dir, f"{video_stem}.json")

        if os.path.exists(blur_metadata_out_path):
            return 0

        cap = cv2.VideoCapture(video_path)
        all_texts = defaultdict(list)
        gt_action = metadata["action"]
        gt_action = gt_action.lower()
        try:
            gt_action_subwords = set(gt_action.split(" "))
            gt_action_subwords.update(set(gt_action.split("-")))
        except Exception as e:
            logger.error(f"Error in splitting the action name: {gt_action}: {e}")
            exit(1)

        counter = 0
        found_relevant_text = False

        while(cap.isOpened()):
            ret, frame = cap.read()
            counter += 1

            if not ret:
                break
 
            if counter % 10 != 0:
                continue

            texts = detect_text(frame)

            if texts:
                for text in texts:
                    desc = text.description.lower()
                    desc_words = set(desc.split())
                    desc_words.update(set(desc.split("-"))) # add words separated by hyphen
                    if any(word in gt_action_subwords for word in desc_words):
                        found_relevant_text = True
                        break
        
        cap.release()

        if not found_relevant_text:
            if "commentbox" not in metadata or "filter" not in metadata["commentbox"]:
                with open(os.path.join(out_dir, f"{video_stem}.json"), "w") as f:
                    json.dump(all_texts, f)
                return 0

        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        tmp_out_dir = Path(f"/tmp/{video_stem}")
        tmp_out_dir.mkdir(parents=True, exist_ok=True)
        tmp_output_video_path = os.path.join(tmp_out_dir, f"{video_stem}.mp4")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            tmp_output_video_path,
            fourcc,
            fps=fps,
            frameSize=(frame_width, frame_height)
        )
        frames_to_blur = dict()
        counter = 0
        while(cap.isOpened()):
            ret, frame = cap.read()
            counter += 1

            if not ret:
                break

            texts = detect_text(frame)
            relevant_texts = []
            if texts:
                for text in texts:
                    desc = text.description
                    desc = ''.join(e for e in desc if e.isalpha())
                    if any(word.lower() in all_action_names_subwords for word in desc.split()):
                        relevant_texts.append({
                            "description": text.description,
                            "bounding_poly": {"vertices": [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]}
                        })

            if relevant_texts:
                frame = obfuscate_text_areas(frame, relevant_texts)
                frames_to_blur[str(counter)] = relevant_texts

            out.write(frame)
        metadata.update({"frames_to_blur": frames_to_blur})
        out.release()
        cap.release()

        final_output_video_path = os.path.join(out_dir, f"{video_stem}.mp4")
        try:
            reencode_video(tmp_output_video_path, final_output_video_path) 
        except Exception as e:
            logger.error(f"Error in reencoding blurred video {tmp_output_video_path}: {e}")

        with open(blur_metadata_out_path, "w") as f:
            json.dump(metadata, f)
 
    data = read_jsonl(jsonl_fpath)

    if debug:
        data = data[:1]

    videos = Path(video_segments_dir).glob("*.mp4")
    video_fstem_to_video_path = {Path(video_path).stem: video_path.as_posix() for video_path in videos}
    for d in data:
        d.update({"video_path": video_fstem_to_video_path[f"{d['youtube_id']}_{d['start_timestamp']}_{d['end_timestamp']}"]})

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        futures = [
            executor.submit(_blur_text_single_video, d) for d in data
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Obfuscating text"):
            result = future.result()


def obfuscate_text_in_videos_with_masks(
    jsonl_fpath: str,
    max_workers: int,
    video_segments_dir: str,
    out_dir: str,
    debug: bool = False
    ):
    """Uses the already generated masks to obfuscate text in the videos."""
    os.makedirs(out_dir, exist_ok=True)
    all_action_names = json.load(open("../data/all_actions.json"))
    all_action_names = [v["action"] for v in all_action_names.values()]
    all_action_names_subwords = set([word.lower() for action in all_action_names for word in action.split()])
    all_action_names_subwords.update({
        "legal", "north", "south", "explosive", "not", "legal",
        "backheel", "!", "trickshot", "faint", "Ganda", "ng",
        "ni", "Idol", "carry!", "fancy", "skills",
    })

    def _blur_text_single_video(
        metadata: Dict
    ):
        """Uses the already generated masks to obfuscate text in the videos."""
        youtube_id = metadata["youtube_id"]
        start = round(float(metadata["start_timestamp"]), 1)
        end = round(float(metadata["end_timestamp"]), 1)

        video_fname = f"{youtube_id}_{start}_{end}.mp4"
        video_path = Path(video_segments_dir) / video_fname

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found at {video_path}")

        frames_to_blur = metadata.get("frames_to_blur", [])
        if not frames_to_blur:
            shutil.copy(video_path, out_dir)
            return 0
        
        cap = cv2.VideoCapture(video_path.as_posix())
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        tmp_out_dir = Path(f"/tmp/{video_path.stem}")
        tmp_out_dir.mkdir(parents=True, exist_ok=True)
        tmp_output_video_path = os.path.join(tmp_out_dir, video_fname)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            tmp_output_video_path,
            fourcc,
            fps=fps,
            frameSize=(frame_width, frame_height)
        )

        counter = 0
        while(cap.isOpened()):
            ret, frame = cap.read()
            counter += 1

            if not ret:
                break

            logger.info(f"{Path(video_path).name}, Frame: {counter}")

            if str(counter) in frames_to_blur:
                texts = frames_to_blur[str(counter)]

                if texts:
                    frame = obfuscate_text_areas(frame, texts)
            
            out.write(frame)

        out.release()
        cap.release()

        final_output_video_path = os.path.join(out_dir, f"{video_fname}")
        cmd = f"ffmpeg -loglevel quiet -i {tmp_output_video_path} -y -c:v libx264 -preset slow -crf 22 -c:a aac -strict -2 {final_output_video_path}"
        out = subprocess.run(cmd, shell=True)
 
    data = read_jsonl(jsonl_fpath)

    if debug:
        data = data[:1]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for d in data:
            futures.append(executor.submit(_blur_text_single_video, d))

        for future in tqdm(as_completed(futures), total=len(futures), desc="Blurring text"):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error in future: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--function", type=str, 
                        choices=["obfuscate_text_in_videos_gcloud_vision_api", "obfuscate_text_in_videos_with_masks"],
                        help="Choose which function to run",
                        default="obfuscate_text_in_videos_with_masks")
    parser.add_argument("--data_fpath", type=str, required=True,
                        help="Path to the jsonl file released by ActionAtlas team containing the metadata.")
    parser.add_argument("--max_workers", type=int, default=1,
                        help="Number of workers to use for multithreading. We recommend using no more"
                        "than 4 workers when using Google Cloud Vision API.")
    parser.add_argument("--video_segments_dir", type=str, required=True,
                        help="Path to the directory containing the extracted segments using extract_segments script.")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Path to the directory to save all the video in ActionAtlas including the ones with obfuscated text.")
    parser.add_argument("--debug", action="store_true", help="If set, the script will run in debug mode on one video.")

    args = parser.parse_args()

    if args.function == "obfuscate_text_in_videos_gcloud_vision_api":
        obfuscate_text_in_videos_gcloud_vision_api(
            jsonl_fpath=args.data_fpath,
            max_workers=args.max_workers,
            video_segments_dir=args.videos_dir,
            out_dir=args.out_dir,
            debug=args.debug
        )
    elif args.function == "obfuscate_text_in_videos_with_masks":
        obfuscate_text_in_videos_with_masks(
            jsonl_fpath=args.data_fpath,
            max_workers=args.max_workers,
            video_segments_dir=args.video_segments_dir,
            out_dir=args.out_dir,
            debug=args.debug
        )