"""
Evaluating proprietary models on benchmarks.
"""
from concurrent.futures import ThreadPoolExecutor
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Union
import json
import base64
import tempfile
import math
from multiprocessing import Pool
from io import BytesIO

import numpy as np
import yaml
import wget
from tqdm import tqdm
import cv2
from PIL import Image
import google.generativeai as genai
from loguru import logger
from openai import OpenAI
from sklearn.utils import resample

from action_atlas.utils import (
    read_jsonl, 
    write_jsonl,
    stream_jsonl,
    resolve_media_path,
)
from action_atlas.prompts import (
    MC_PROMPT,
    MC_COT_PROMPT,
    MC_WITH_DESCRIPTION_PROMPT,
    MC_WITH_DESCRIPTION_COT_PROMPT,
    MC_STEP_BY_STEP_ACROSS_FRAMES_PROMPT,
)


def handle_gemini_error(response, e):
    error = f"Failed to generate content. {e}"
    if isinstance(e, IndexError) and len(response.parts) == 0:
        if hasattr(response, "candidates"):
            response = response.candidates[0]
            if hasattr(response, "safety_ratings"):
                error = {"SAFETY": {}}
                for safety_rating in response.safety_ratings:
                    error["SAFETY"][safety_rating.category.name] = safety_rating.probability.name
                error = "Failed to generate content. " + json.dumps(error)
    elif hasattr(response, "prompt_feedback"):
        error = f"Failed to generate content. PROMPT_FEEDBACK: {response.prompt_feedback}"

    return error


def eval_gemini_video_mode(
    action_atlas_jsonl_fpath: str,
    videos_dir: str,
    out_fpath: str,
    model_name: str,
    with_choice_descriptions: bool=False,
    template_prompt: str=MC_PROMPT,
    debug: bool=False,
    ):
    """Evaluates gemini model by uploading a video file to API.
    Args:
        action_atlas_jsonl_fpath (str): Path to the JSONL file containing the ActionAtlas data.
        videos_dir (str): Directory containing the video files.
        out_fpath (str): Path to the output JSONL file.
        model_name (str): Name of the Gemini model to evaluate.
        with_choice_descriptions (bool): whether to include description of each choice.
        template_prompt (str): Template prompt for the model.
        debug (bool): Flag to enable debug mode (process only the first item).

    According to the API documentat, here are the available models:
        - models/gemini-1.0-pro
        - models/gemini-1.0-pro-001
        - models/gemini-1.0-pro-latest
        - models/gemini-1.0-pro-vision-latest
        - models/gemini-1.5-flash
        - models/gemini-1.5-flash-001
        - models/gemini-1.5-flash-latest
        - models/gemini-1.5-pro
        - models/gemini-1.5-pro-001
        - models/gemini-1.5-pro-latest
        - models/gemini-pro
        - models/gemini-pro-vision

    Ideally, this function should be multithreaded to make it faster.
    However, my ip got banned when I tried that.
    """
    key = os.getenv("GEMINI_API_KEY")
    genai.configure(api_key=key)
    logger.info(f"Evaluating {model_name} in video mode")
    model = genai.GenerativeModel(model_name=model_name)

    data = read_jsonl(action_atlas_jsonl_fpath)
    if debug:
        data = data[:1]
    
    fix_errors_only = False
    if os.path.exists(out_fpath):
        logger.info(f"Output file {out_fpath} already exists. Running in fix errors only mode...")
        res = read_jsonl(out_fpath)
        fix_errors_only = True

    all_results = []
    for idx, d in enumerate(tqdm(data)):
        try:
            youtube_id = d["youtube_id"]
            start = round(float(d["start_timestamp"]), 1)
            end = round(float(d["end_timestamp"]), 1)
            video_path = Path(videos_dir) / f"{youtube_id}_{start}_{end}.mp4"

            question = d["question"]
            choices = d["choices_str"]

            if with_choice_descriptions:
                choices = choices.split("\n")
                descs :List = d["choice_descriptions"]
                assert len(choices) == len(descs)
                choices = [f"{o}: {desc}" for o, desc in zip(choices, descs)]
                choices = "\n".join(choices)

            if fix_errors_only:
                if not "error" in res[idx]:
                    all_results.append(res[idx])
                    continue

            video_file = genai.upload_file(path=video_path)

            while video_file.state.name == "PROCESSING":
                print('.', end='')
                time.sleep(10)
                video_file = genai.get_file(video_file.name)

            if video_file.state.name == "FAILED":
                logger.error(f"Failed to upload {video_path}.")
                all_results.append({"video_path": video_path, "question":question, "choices": choices, "gt_answer": d["answer"], "response": "-1", "id": d["id"], "error": "Failed to upload video."})
                continue
        except Exception as e:
            logger.error(f"Failed to process {video_path}. {e}. Skipping...")
            all_results.append({"video_path": video_path, "question": question, "choices": choices, "gt_answer": d["answer"], "response": "-1", "id": d["id"], "error": f"Failed to process video. {e}"})
            continue

        prompt = template_prompt.format(question=question, choices=choices)
        try:
            response = model.generate_content(
                [prompt, video_file],
                request_options={"timeout": 600}
            )
            response = response.parts[0].text.strip().replace(".", "")
            all_results.append({"video_path": video_path, "question": question, "choices": choices, "gt_answer": d["answer"], "id": d["id"], "response": response})
        except Exception as e:
            error = handle_gemini_error(response, e)
            all_results.append({"video_path": video_path, "question":question, "choices": choices, "gt_answer": d["answer"], "id": d["id"], "response": "-1", "error": error})
            logger.error(f"Failed to generate content for {video_path}. {error}")

    write_jsonl(all_results, out_fpath)


def eval_gemini_image_mode_fixed_fps(
    action_atlas_jsonl_fpath: str,
    out_fpath: str,
    videos_dir: str,
    model_name: str,
    fps: int,
    with_choice_descriptions: bool=False,
    template_prompt: str=MC_PROMPT,
    debug: bool=False,
):
    """
    Evaluate a Gemini model in image mode with a fixed FPS.

    Args:
        action_atlas_jsonl_fpath (str): Path to the JSONL file containing the ActionAtlas data.
        out_fpath (str): Path to the output JSONL file.
        videos_dir (str): Directory containing the video files.
        model_name (str): Name of the Gemini model to evaluate.
        with_choice_descriptions (bool): whether to include description of each choice.
        fps (int): Frames per second to extract images from the video.
        debug (bool): Flag to enable debug mode (process only the first item).

    Returns:
        None
    """
    key = os.getenv("GEMINI_API_KEY")
    genai.configure(api_key=key)
    logger.info(f"Evaluating {model_name} with fps {fps}")
    model = genai.GenerativeModel(model_name=model_name)

    data = read_jsonl(action_atlas_jsonl_fpath)
    if debug:
        data = data[:1]

    fix_errors_only = False
    if os.path.exists(out_fpath):
        logger.info(f"Output file {out_fpath} already exists. Running in fix errors only mode...")
        res = read_jsonl(out_fpath)
        fix_errors_only = True

    all_results = []
    for idx, d in enumerate(tqdm(data)):
        try:
            youtube_id = d["youtube_id"]
            start = round(float(d["start_timestamp"]), 1)
            end = round(float(d["end_timestamp"]), 1)
            video_path = (Path(videos_dir) / f"{youtube_id}_{start}_{end}.mp4").as_posix()

            question = d["question"]
            choices = d["choices_str"]

            if with_choice_descriptions:
                choices = choices.split("\n")
                descs :List = d["choice_descriptions"]
                assert len(choices) == len(descs)
                choices = [f"{o}: {desc}" for o, desc in zip(choices, descs)]
                choices = "\n".join(choices)

            if fix_errors_only:
                if not "error" in res[idx]:
                    all_results.append(res[idx])
                    continue

            video = cv2.VideoCapture(video_path)
            question = d["question"]

            pil_frames = []
            original_fps = video.get(cv2.CAP_PROP_FPS)
            frame_interval = int(original_fps / fps)
 
            frame_count = 0

            while True:
                ret, frame = video.read()
                if not ret:
                    break

                # Check if the current frame is to be sampled
                if frame_count % frame_interval == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Convert the frame to a PIL image
                    pil_frame = Image.fromarray(frame_rgb)
                    pil_frames.append(pil_frame)

                frame_count += 1
            video.release()
        except Exception as e:
            logger.error(f"Failed to process {video_path}. {e}. Skipping...")
            all_results.append({"video_path": video_path, "question": question, "choices": choices, "gt_answer": d["answer"], "response": "-1", "id": d["id"], "error": f"Failed to process video. {e}"})
            continue

        prompt = template_prompt.format(question=question, choices=choices)
        try:
            response = model.generate_content(
                [prompt] + pil_frames,
                request_options={"timeout": 600}
            )
            response = response.parts[0].text.strip().replace(".", "")
            all_results.append({"video_path": video_path, "question": question, "choices": choices, "gt_answer": d["answer"], "id": d["id"], "response": response})
        except Exception as e:
            error = handle_gemini_error(response, e)
            logger.error(f"Failed to generate content for {video_path}. {error}")
            all_results.append({"video_path": video_path, "question":question, "choices": choices, "gt_answer": d["answer"], "id": d["id"], "response": "-1", "error": error})

    write_jsonl(all_results, out_fpath)


def eval_gemini_image_mode_fixed_nframes(
    action_atlas_jsonl_fpath: str,
    out_fpath: str,
    videos_dir: str,
    model_name: str,
    n_sample_frames: int,
    with_choice_descriptions: bool=False,
    template_prompt: str=MC_PROMPT,
    debug: bool=False,
):
    """
    Evaluate a Gemini model in image mode with a fixed number of frames.

    Args:
        action_atlas_jsonl_fpath (str): Path to the JSONL file containing the ActionAtlas data.
        out_fpath (str): Path to the output JSONL file.
        videos_dir (str): Directory containing the video files.
        model_name (str): Name of the Gemini model to evaluate.
        n_sample_frames (int): Number of sample frames to use for evaluation.
        with_choice_descriptions (bool): whether to include description of each choice.
        template_prompt (str): Template prompt for the model.
        debug (bool): Flag to enable debug mode (process only the first item).

    Returns:
        None
    """
    key = os.getenv("GEMINI_API_KEY")
    genai.configure(api_key=key)
    logger.info(f"Evaluating {model_name} with n_frames {n_sample_frames}")
    model = genai.GenerativeModel(model_name=model_name)

    data = read_jsonl(action_atlas_jsonl_fpath)

    all_results = []

    if debug:
        data = data[:1]

    fix_errors_only = False
    if os.path.exists(out_fpath):
        logger.info(f"Output file {out_fpath} already exists. Running in fix errors only mode...")
        res = read_jsonl(out_fpath)
        fix_errors_only = True

    for idx, d in enumerate(tqdm(data)):
        try:
            youtube_id = d["youtube_id"]
            start = round(float(d["start_timestamp"]), 1)
            end = round(float(d["end_timestamp"]), 1)
            video_path = (Path(videos_dir) / f"{youtube_id}_{start}_{end}.mp4").as_posix()

            question = d["question"]
            choices = d["choices_str"]

            if with_choice_descriptions:
                choices = choices.split("\n")
                descs :List = d["choice_descriptions"]
                assert len(choices) == len(descs)
                choices = [f"{o}: {desc}" for o, desc in zip(choices, descs)]
                choices = "\n".join(choices)

            if fix_errors_only:
                if not "error" in res[idx]:
                    all_results.append(res[idx])
                    continue

            video = cv2.VideoCapture(video_path)

            pil_frames = []
            frame_count = 0

            while True:
                ret, frame = video.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Convert the frame to a PIL image
                pil_frame = Image.fromarray(frame_rgb)
                pil_frames.append(pil_frame)

                frame_count += 1
            video.release()

            if n_sample_frames > 1:
                inds = np.linspace(0, len(pil_frames) - 1, n_sample_frames, dtype=int)
            else:
                # get the middle frame
                inds = [len(pil_frames) // 2]
            pil_frames = [pil_frames[i] for i in inds]
        except Exception as e:
            logger.error(f"Failed to process {video_path}. {e}. Skipping...")
            all_results.append({"video_path": video_path, "question": question, "choices": choices, "gt_answer": d["answer"], "response": "-1", "id": d["id"], "error": f"Failed to process video. {e}"})
            continue

        prompt = template_prompt.format(question=question, choices=choices)
        try:
            response = model.generate_content(
                [prompt] + pil_frames,
                request_options={"timeout": 600}
            )
            response = response.parts[0].text.strip().replace(".", "")
            all_results.append({"video_path": video_path, "question": question, "choices": choices, "id": d["id"], "gt_answer": d["answer"], "response": response})
        except Exception as e:
            error = handle_gemini_error(response, e)
            logger.error(f"Failed to generate content for {video_path}. {error}")
            all_results.append({"video_path": video_path, "question":question, "choices": choices, "gt_answer": d["answer"], "response": "-1", "id": d["id"], "error": error})

    write_jsonl(all_results, out_fpath)


def eval_gpt4_fixed_nframes(
    model: str,
    out_fpath: str,
    n_sample_frames: int,
    action_atlas_jsonl_fpath: str,
    template_prompt: str,
    videos_dir: str,
    with_choice_descriptions: bool = False,
    detail: str = "low",
    debug: bool = False,
) -> None:
    """
    Evaluate a GPT-4 model with a fixed number of frames.

    Args:
        model (str): The name of the GPT-4 model to evaluate.
        out_fpath (str): Path to the output JSONL file.
        n_sample_frames (int): Number of sample frames to use for evaluation.
        action_atlas_jsonl_fpath (str): Path to the JSONL file containing the ActionAtlas data.
        template_prompt (str): Template prompt for the model.
        videos_dir (str): Directory containing the video files.
        with_choice_descriptions (bool): whether to include description of each choice.
        detail (str): Level of detail for the evaluation (default is "low").
        debug (bool): Flag to enable debug mode (process only the first item).

    Returns:
        None
    """
    key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=key)
    data = read_jsonl(action_atlas_jsonl_fpath)

    if debug:
        data = data[:1]

    all_results = []
    logger.info(f"Evaluating {model} with {n_sample_frames} frames detail {detail}")

    fix_errors_only = False
    if os.path.exists(out_fpath):
        res = read_jsonl(out_fpath)
        fix_errors_only = True
        logger.info("Running in fix errors mode only.")

    for idx, d in enumerate(tqdm(data)): 
        youtube_id = d["youtube_id"]
        start = round(float(d["start_timestamp"]), 1)
        end = round(float(d["end_timestamp"]), 1)
        video_path = (Path(videos_dir) / f"{youtube_id}_{start}_{end}.mp4").as_posix()

        question = d["question"]
        choices = d["choices_str"]

        if with_choice_descriptions:
            choices = choices.split("\n")
            descs :List = d["choice_descriptions"]
            assert len(choices) == len(descs)
            choices = [f"{o}: {desc}" for o, desc in zip(choices, descs)]
            choices = "\n".join(choices)

        if fix_errors_only:
            if not "error" in res[idx]:
                all_results.append(res[idx])
                continue
        try:
            video = cv2.VideoCapture(video_path)
            base64Frames = []

            while video.isOpened():
                success, frame = video.read()
                if not success:
                    break
                _, buffer = cv2.imencode(".jpg", frame)
                base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

            video.release()

            if n_sample_frames > 1:
                inds = np.linspace(0, len(base64Frames) - 1, n_sample_frames, dtype=int)
            else:
                # get the middle frame
                inds = [len(base64Frames) // 2]

            base64Frames = [base64Frames[i] for i in inds]

            messages=[
                {"role":"system", "content":"You're an assistant great at recognizing fine-grained actions and activities in a video."},
                {"role":"user", "content":[
                    f"These are frames extracted from a video. {template_prompt.format(question=question, choices=choices)}",
                    *map(lambda x:{"type":"image_url",
                                    "image_url":{"url":f'data:image/jpg;base64,{x}',"detail":detail}}, base64Frames)
                    ],
                }
            ]
            params = {
                "model": model,
                "messages": messages,
                "max_tokens": 4096,
            }
        except Exception as e:
            logger.error(f"Failed to process {video_path}. {e}. Skipping...")
            all_results.append({"video_path": video_path, "question": question, "choices": choices, "gt_answer": d["answer"], "id": d["id"], "response": -1, "error": f"Failed to process video. {e}"})
            continue

        try:
            result = client.chat.completions.create(**params)
            response = result.choices[0].message.content.strip()
            all_results.append({"video_path": video_path, "question": question, "choices": choices, "gt_answer": d["answer"], "id": d["id"], "response": response})
        except Exception as e:
            logger.error(f"Failed to generate content for {video_path}. {e}")
            all_results.append({"video_path": video_path, "question": question, "choices": choices, "gt_answer": d["answer"], "id": d["id"], "response": -1, "error": f"Failed to generate content. {e}"})

    write_jsonl(all_results, out_fpath)


def eval_gpt4_fixed_fps(
    model,
    fps: int,
    out_fpath: str,
    action_atlas_jsonl_fpath: str,
    template_prompt: str,
    videos_dir: str,
    detail: str="low",
    with_choice_descriptions: bool=False,
    max_n_frames: int=100,
    debug: bool=False,
    ):
    """
    Evaluate a GPT-4 model with a fixed frames per second (FPS).

    Args:
        model (str): The name of the GPT-4 model to evaluate.
        fps (int): Frames per second to extract images from the video.
        out_fpath (str): Path to the output JSONL file.
        action_atlas_jsonl_fpath (str): Path to the JSONL file containing the ActionAtlas data.
        template_prompt (str): Template prompt for the model.
        videos_dir (str): Directory containing the video files.
        detail (str): Level of detail for the evaluation (default is "low").
        with_choice_descriptions (bool): Whether to include description of each choice.
        max_n_frames (int): Maximum number of frames to process.
        debug (bool): Flag to enable debug mode (process only the first item).

    Returns:
        None
    """
    key = os.getenv("OPENAI_API_KEY")
    logger.info(f"Evaluating {model} with fps {fps}, {detail} detail, max frames {max_n_frames}")
    client = OpenAI(api_key=key)
    data = read_jsonl(action_atlas_jsonl_fpath)
    all_results = []

    if debug:
        data = data[:1]

    fix_errors_only = False
    if os.path.exists(out_fpath):
        logger.warning("Running in fix errors only mode")
        res = read_jsonl(out_fpath)
        fix_errors_only = True

    for idx, d in enumerate(tqdm(data)):
        if fix_errors_only:
            if not "error" in res[idx]:
                all_results.append(res[idx])
                continue

        youtube_id = d["youtube_id"]
        start = round(float(d["start_timestamp"]), 1)
        end = round(float(d["end_timestamp"]), 1)
        video_path = (Path(videos_dir) / f"{youtube_id}_{start}_{end}.mp4").as_posix()

        question = d["question"]
        choices = d["choices_str"]

        if with_choice_descriptions:
            choices = choices.split("\n")
            descs :List = d["choice_descriptions"]
            assert len(choices) == len(descs)
            choices = [f"{o}: {desc}" for o, desc in zip(choices, descs)]
            choices = "\n".join(choices)

        try:
            video = cv2.VideoCapture(video_path)

            base64Frames = []
            original_fps = video.get(cv2.CAP_PROP_FPS)
            frame_interval = int(original_fps / fps)
            
            frame_count = 0

            while True:
                ret, frame = video.read()
                if not ret:
                    break
                if frame_count % frame_interval == 0:
                    _, buffer = cv2.imencode(".jpg", frame)
                    base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
                frame_count += 1

            video.release()

            if len(base64Frames) > max_n_frames:
                logger.info(f"Downsampling {len(base64Frames)} frames to {max_n_frames} for video {video_path}")
                base64Frames = base64Frames[::len(base64Frames) // max_n_frames]
                base64Frames = base64Frames[:max_n_frames]

            messages=[
                {"role":"system","content":"You're an assistant great at recognizing fine-grained actions and activities in a video."},
                {"role":"user", "content":[
                    f"These are frames extracted from a video. {template_prompt.format(question=question, choices=choices)}",
                    *map(lambda x:{"type":"image_url",
                                    "image_url":{"url":f'data:image/jpg;base64,{x}',"detail":detail}}, base64Frames)
                    ],
                }
            ]
            params = {
                "model": model,
                "messages": messages,
                "max_tokens": 4096,
            }
        except Exception as e:
            logger.error(f"Failed to process {video_path}. {e}. Skipping...") 
            all_results.append({"video_path": video_path, "question": question, "choices": choices, "gt_answer": d["answer"], "response": -1, "id": d["id"], "error": f"Failed to process video. {e}"})
            continue

        try:
            import pdb; pdb.set_trace()
            result = client.chat.completions.create(**params)
            response = result.choices[0].message.content.strip().replace(".", "")
            all_results.append({"video_path": video_path, "question": question, "choices": choices, "gt_answer": d["answer"], "response": response, "id": d["id"], "n_frames": len(base64Frames)})
        except Exception as e:
            logger.error(f"Failed to generate content for {video_path}. {e}") 
            all_results.append({"video_path": video_path, "question": question, "choices": choices, "gt_answer": d["answer"], "response": -1, "error": f"Failed to generate content. {e}", "id": d["id"], "n_frames": len(base64Frames)})
 
    write_jsonl(all_results, out_fpath)


def bootstrap_confidence_interval_accuracy(
    y_true, 
    y_pred, 
    num_bootstrap=1000):
    """
    Calculate 95% confidence interval for the overall accuracy using the bootstrap method.

    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Model predicted labels.
    num_bootstrap (int): Number of bootstrap samples to use for calculating confidence intervals.

    Returns:
    tuple: (lower_bound, upper_bound) for the 95% confidence interval of the accuracy.
    """
    n = len(y_pred)
    bootstrap_accuracies = np.zeros(num_bootstrap)
    for i in range(num_bootstrap):
        # Resample indices
        indices = np.random.choice(range(n), size=n, replace=True)
        # Resample true and predicted labels
        y_true_bootstrap = y_true[indices]
        y_pred_bootstrap = y_pred[indices]
 
        # Calculate accuracy for the resampled data
        bootstrap_accuracies[i] = np.mean(y_pred_bootstrap == y_true_bootstrap)

    # Calculate the 95% confidence interval for accuracy
    lower_bound = np.percentile(bootstrap_accuracies, 2.5)
    upper_bound = np.percentile(bootstrap_accuracies, 97.5)

    mean_accuracy = np.mean(bootstrap_accuracies)
    margin_of_error = mean_accuracy - lower_bound

    plus_minus = u"\u00B1"
    logger.info(f"Mean acc. with margin of error: {mean_accuracy * 100:.2f} {plus_minus} {margin_of_error*100:.2f}")
    logger.info(f"95% Confidence Interval: [{lower_bound}, {upper_bound}]")

    return mean_accuracy, lower_bound, upper_bound, margin_of_error


def compute_final_accuracy(
    result_jsonl_fpath: str, 
    verbose: bool=False, 
    conf_interval: bool=False
):
    results = read_jsonl(result_jsonl_fpath)
    errors = 0
    if conf_interval:
        y_true = []
        y_pred = []

    for it, r in enumerate(results, 1):
        if "error" in r:
            r["correct"] = 0
            errors += 1
            response = "-1"
            if verbose:
                logger.info(f"{it}. ground truth: {str(r['gt_answer'])}, response: {response}")
        else:
            response = str(r["response"]).strip() 
            response = "".join(filter(str.isdigit, response)) # remove any non digit characters
            if len(response) > 1:
                response = response[0]
            if verbose:
                logger.info(f"{it}. ground truth: {str(r['gt_answer'])}, response: {response}")

        if not conf_interval:
            if response == str(r["gt_answer"]):
                r["correct"] = 1
            else:
                r["correct"] = 0
        else:
            y_true.append(str(r["gt_answer"]))
            y_pred.append(response)
    
    logger.info(f"result_jsonl_fpath: {result_jsonl_fpath}") 
    logger.info(f"Errors: {errors}")

    if not conf_interval:    
        acc = sum([r['correct'] for r in results]) / len(results)
        logger.info(f"Accuracy for {result_jsonl_fpath}: {acc}")

        return acc

    res = bootstrap_confidence_interval_accuracy(
        y_true=np.array(y_true),
        y_pred=np.array(y_pred),
    )

    return res


def compute_final_accuracy_with_cot_reasoning(
    result_jsonl_fpath: str, 
    verbose: bool=False, 
    conf_interval: bool=False):
    results = read_jsonl(result_jsonl_fpath)

    if conf_interval:
        y_true = []
        y_pred = []

    for it, r in enumerate(results, 1):
        if "error" in r:
            r["correct"] = 0
            errors += 1
            response = "-1"
            if verbose:
                logger.info(f"{it}. ground truth: {str(r['gt_answer'])}, response: {response}") 
        else:
            response = str(r["response"]).split("\n")[-1]
            # remove any non digit characters
            response = "".join(filter(str.isdigit, response))
            if len(response) > 1:
                response = response[0]
            elif len(response) == 0:
                # Check if the model has outputted only one of the action names
                choices = r["choices"]
                if isinstance(choices, str):
                    choices = choices.split("\n")
                    choices = [opt[len("1. "):].lower() for opt in choices]  # strip off the choice index
                response = str(r["response"]).lower()
                found_option_idx = None
                for idx, opt in enumerate(choices, 1):
                    if opt in response:
                        if found_option_idx is None:
                            found_option_idx = idx
                        else:
                            found_option_idx = None
                            break
                response = "" if found_option_idx is None else str(found_option_idx)
                # last resort: check if only a single digit exists in the response which we assume is the answer
                if len(response) == 0:
                    response = str(r["response"])
                    # remove any non digit characters
                    response = "".join(filter(str.isdigit, response))
                    # check if the response is a single digit
                    if len(response) == 1:
                        response = response[0]
                    else:
                        response = ""
 
        if verbose:
            logger.info(f"{it}. ground truth: {str(r['gt_answer'])}, response: {response}")
        
        if not conf_interval:
            if response == str(r["gt_answer"]):
                r["correct"] = 1
            else:
                r["correct"] = 0
        else:
            y_true.append(str(r["gt_answer"]))
            y_pred.append(response)
    
    if not conf_interval:    
        acc = sum([r['correct'] for r in results]) / len(results)
        logger.info(f"Accuracy: {acc}")
        return acc

    res = bootstrap_confidence_interval_accuracy(
        y_true=np.array(y_true),
        y_pred=np.array(y_pred),
    )

    return res


def compute_gemini_refusal_rate(preds_fpath: Union[str, Path]):
    """Gemini classifies many of the data points as harmful and returns
    the safety ratings. This function calculates the refusal rates based
    on that.
    
    should check `response.prompt_feedback` in returned responses as well.
    """
    preds = read_jsonl(preds_fpath)
    safety_refusals = 0
    total_errors = 0

    for pred in preds:
        if "error" in pred:
            if "SAFETY" in pred["error"]:
                safety_refusals += 1
        if pred["response"] == "-1":
            total_errors +=1

    logger.info(f"Preds fpath: {preds_fpath}") 
    logger.info(f"Number of safety refusals: {safety_refusals} ({(safety_refusals / len(preds)) * 100:.2f}%)") 
    logger.info(f"Toal errors: {total_errors} ({(total_errors / len(preds)) * 100:.2f}%)")


if __name__ == "__main__":
    ## gpt4 fixed fps
    # model = "gpt-4o"; n_frames=2; detail="low"
    # eval_gpt4_fixed_nframes(
    #     model=model,
    #     n_sample_frames=n_frames,
    #     out_fpath=f"../data/action_atlas_v1_{model}_{n_frames}frames_{detail}_detail_.jsonl",
    #     action_atlas_jsonl_fpath="../data/action_atlas_v1.jsonl",
    #     videos_dir="../data/blurred_segments",
    #     template_prompt=MC_PROMPT,
    #     detail=detail
    # )
    ##
    model = "gpt-4o"; fps=2; detail="low"
    eval_gpt4_fixed_fps(
        model=model,
        fps=fps,
        out_fpath=f"../data/action_atlas_v1_{model}_{fps}fps_{detail}_detail_.jsonl",
        action_atlas_jsonl_fpath="../data/action_atlas_v1.jsonl",
        videos_dir="../data/blurred_segments",
        template_prompt=MC_PROMPT,
        with_choice_descriptions=True,
        detail=detail
    )
    ## gemini video mode
    # api_model_name = "models/gemini-1.5-pro-latest"
    # model_name = api_model_name.replace("models/", "")
    # eval_gemini_video_mode(
    #     model_name=api_model_name,
    #     videos_dir="../data/blurred_segments",
    #     out_fpath=f"../data/{model_name}_action_atlas_v1_video_mode_preds.jsonl",
    #     action_atlas_jsonl_fpath="../data/action_atlas_v1.jsonl",
    #     template_prompt=MC_PROMPT,
    # )
    ## gemini image mode fixed fps
    # api_model_name = "models/gemini-1.5-pro-latest"; fps=2
    # model_name = api_model_name.replace("models/", "")
    # eval_gemini_image_mode_fixed_fps(
    #     model_name=api_model_name,
    #     fps=fps,
    #     videos_dir="../data/blurred_segments",
    #     out_fpath=f"../data/{model_name}_action_atlas_v1_image_mode_{fps}fps_preds.jsonl",
    #     action_atlas_jsonl_fpath="../data/action_atlas_v1.jsonl",
    #     template_prompt=MC_PROMPT,
    # )
    ##
    # api_model_name = "models/gemini-1.5-pro-latest"; n_frames=2
    # model_name = api_model_name.replace("models/", "")
    # eval_gemini_image_mode_fixed_nframes(
    #     model_name=api_model_name,
    #     n_sample_frames=2,
    #     videos_dir="../data/blurred_segments",
    #     out_fpath=f"../data/{model_name}_action_atlas_v1_image_mode_{n_frames}frames_preds.jsonl",
    #     action_atlas_jsonl_fpath="../data/action_atlas_v1.jsonl",
    #     template_prompt=MC_PROMPT,
    # )
    ##