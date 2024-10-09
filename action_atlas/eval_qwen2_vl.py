import json
from typing import List, Union
from pathlib import Path
import argparse
import os

import torch
from loguru import logger
from tqdm import tqdm
import wget
import numpy as np
import cv2
from PIL import Image
import base64

from fvcore.nn import FlopCountAnalysis

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info


def read_jsonl(path: Union[str, Path]) -> List[dict]:
    lines = []
    with open(path, 'r') as jsonl_file:
        for line in jsonl_file:
            lines.append(json.loads(line))
    
    return lines


def write_jsonl(data, path):
    with open(path, 'w') as jsonl_file:
        for line in data:
            json_str = json.dumps(line)
            jsonl_file.write(json_str + '\n')


MC ="""Answer the given question according to the video. Only output the choice number and nothing else. When answering the question consider all legal and illegal moves and drills.\n{question}\n{options}"""
MC_WITH_DESCRIPTION ="""Answer the given question according to the video. Each option presents an action name along with its description which helps in identification. Only output the choice number and nothing else. When answering the question consider all legal and illegal moves and drills.\n{question}\n{options}"""
MC_WITH_DESCRIPTION_COT ="""Answer the given question according to the video. Each option presents an action name along with its description which helps in identification. You can describe the video or do any step by step reasoning about what you see in the video and the options. However, output your final choice number (1 to 5) at the end of your response in a new line. When answering the question consider all legal and illegal moves and drills.\n{question}\n{options}"""
MC_COT = """Answer the given question according to the video. You can describe the video or do any step by step reasoning about what you see in the video. However, it's CRUCIAL TO OUTPUT YOUR FINAL CHOICE NUMBER (1 to 5) AT THE END OF YOUR RESPONSE IN A NEW LINE.\n{question}\n{options}"""


def load_model(attn_implementation:str = "eager"):
    # default: Load the model on the available device(s)
    assert attn_implementation in ["eager", "flash_attention_2", "sdpa"]
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        # torch_dtype="auto",
        attn_implementation=attn_implementation,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    # model = Qwen2VLForConditionalGeneration.from_pretrained(
    #     "Qwen/Qwen2-VL-7B-Instruct",
    #     torch_dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2",
    #     device_map="auto",
    # )

    return model


# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)


def bootstrap_confidence_interval_accuracy(y_true, y_pred, num_bootstrap=1000):
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
    # logger.info(f"Mean Accuracy: {mean_accuracy}")
    # logger.info(f"Margin of Error: {margin_of_error}")
    logger.info(f"95% Confidence Interval: [{lower_bound}, {upper_bound}]")

    return mean_accuracy, lower_bound, upper_bound, margin_of_error


def compute_final_accuracy(result_jsonl_fpath: str, verbose: bool=False, conf_interval: bool=False):
    """
    """
    results = read_jsonl(result_jsonl_fpath)
    errors = 0
    if conf_interval:
        y_true = []
        y_pred = []

    for r in results:
        if "error" in r:
            r["correct"] = 0
            errors += 1
        response = str(r["response"]).strip() 
        response = "".join(filter(str.isdigit, response)) # remove any non digit characters
        if len(response) > 1:
            response = response[0]
        if verbose:
            logger.info(f"ground truth: {str(r['gt_answer'])}, response: {response}")
        # print(response, str(r["gt_answer"]))
        # print(response)
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


def compute_final_accuracy_cot(result_jsonl_fpath: str, verbose: bool=False, conf_interval: bool=False):
    results = read_jsonl(result_jsonl_fpath)

    if conf_interval:
        y_true = []
        y_pred = []

    for it, r in enumerate(results, 1):
        response = str(r["response"]).split("\n")[-1]
        # remove any non digit characters
        response = "".join(filter(str.isdigit, response))
        if len(response) > 1:
            response = response[0]
        elif len(response) == 0:
            # Check if the model has outputted only one of the action names
            options = r["options"]
            if isinstance(options, str):
                options = options.split("\n")
                options = [opt[len("1. "):].lower() for opt in options]  # strip off the choice index
            response = str(r["response"]).lower()
            found_option_idx = None
            for idx, opt in enumerate(options, 1):
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


def video_eval_nframes(
    benchmark_jsonl_fpath: str,
    template_prompt: str,
    out_fpath: str,
    with_description: bool = False,
    n_sample_frames: int = 1,
):
    """Eval Qwen2-VL-7B on with a uniformly sampled set of frames."""
    model = load_model(attn_implementation="eager")
    data = read_jsonl(benchmark_jsonl_fpath)
    all_results = []

    if os.path.exists(out_fpath):
        logger.info(f"Output file exists. Returning...")
        return

    if with_description:
        logger.info("Evaluating with description.")

    # starts from the largest size and decreases if the model fails to process the video
    sizes_to_try = [(28 * i) * (28 * i) for i in range(12, 7, -1)]
    
    for idx, d in enumerate(tqdm(data)):
        video_path = d["video_path"]
        question = d["question"]
        options = d["options"]
        local_video_path = f"/tmp/{Path(video_path).name}"

        if with_description:
            options = options.split("\n")
            descs = d["option_descs"].split("\n\n")
            assert len(options) == len(descs)
            options = [f"{o}: {desc}" for o, desc in zip(options, descs)]
            options = "\n".join(options)

        try:
            wget.download(video_path, out=local_video_path, bar=None)
        except Exception as e:
            logger.error(f"Failed to process {video_path}. {e}. Skipping...") 

        prompt = template_prompt.format(options=options, question=question)

        video = cv2.VideoCapture(local_video_path)
        question = d["question"]

        pil_frames = []
        frame_count = 0

        duration = video.get(cv2.CAP_PROP_POS_MSEC)
        fps = video.get(cv2.CAP_PROP_FPS)

        while True:
            ret, frame = video.read()
            if not ret:
                break
            # Check if the current frame is to be sampled
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert the frame to a PIL image
            pil_frame = Image.fromarray(frame_rgb)
            pil_frames.append(pil_frame)

            frame_count += 1
        video.release()

        assert n_sample_frames > 1, "Use n_samples_frames > 1 as Qwen2 VL extrapolates it in process_vision_info."
        inds = np.linspace(0, len(pil_frames) - 1, n_sample_frames, dtype=int)

        pil_frames = [pil_frames[i] for i in inds]

        # compute fps for the sampled frames
        # sampled_fps = fps * len(pil_frames) / frame_count
        
        # write pil frames to files
        for i, pil_frame in enumerate(pil_frames):
            pil_frame.save(f"/tmp/{Path(video_path).stem}_{i}.jpg")

        for size in sizes_to_try:
            try:
                messages = [
                {
                    "role": "user",
                    # not gonna use resized_height and width as it naively resizes the image 
                    # into a square which can distort the aspect ratio
                    "content": [
                        {
                            "type": "video",
                            "video": [f"/tmp/{Path(video_path).stem}_{i}.jpg" for i in range(len(pil_frames))],
                            # "fps": sampled_fps,
                            "nframes": len(pil_frames),
                            "max_pixels": size,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }] 
                
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                logger.info(f"input shape: {inputs.input_ids.shape}, size: {size}")
                inputs = inputs.to("cuda")

                # Inference
                generated_ids = model.generate(**inputs, max_new_tokens=512)
                
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )

                if isinstance(output_text, list):
                    output_text = output_text[0]
                
                logger.info(f"Number of frames: {len(pil_frames)}, Output text: {output_text}")
                all_results.append({"video_path": video_path, "question": question, "options": options, "gt_answer": d["answer"], "response": output_text, "id": d["id"], "size": size, "n_sample_frames": n_sample_frames})
                break
            except Exception as e:
                logger.error(f"Failed to process {video_path}. {e}. Skipping...")
                if size == sizes_to_try[-1]:
                    all_results.append({"video_path": video_path, "question": question, "options": options, "gt_answer": d["answer"], "response": "-1", "id": d["id"], "error": str(e)}) 
                else:
                    logger.info(f"Retrying with next smaller size...")

    write_jsonl(all_results, out_fpath)


def video_eval_fps(
    benchmark_jsonl_fpath: str,
    template_prompt: str,
    out_fpath: str,
    fps: float = 1.0,
    with_description: bool = False,
    ):
    if os.path.exists(out_fpath):
        logger.info(f"Output file exists. Returning...")
        return

    model = load_model()
    data = read_jsonl(benchmark_jsonl_fpath)
    all_results = []

    if with_description:
        logger.info("Evaluating with description.")

    # try multiplies of 28 starting from 336 down to 224
    sizes_to_try = [(28 * i) * (28 * i) for i in range(12, 7, -1)]
    
    for idx, d in enumerate(tqdm(data)):
        video_path = d["video_path"]
        question = d["question"]
        options = d["options"]
        local_video_path = f"/tmp/{Path(video_path).name}"

        if with_description:
            options = options.split("\n")
            descs = d["option_descs"].split("\n\n")
            assert len(options) == len(descs)
            options = [f"{o}: {desc}" for o, desc in zip(options, descs)]
            options = "\n".join(options)

        try:
            wget.download(video_path, out=local_video_path, bar=None)
        except Exception as e:
            logger.error(f"Failed to download {video_path}. {e}. Skipping...")
            continue

        prompt = template_prompt.format(options=options, question=question)

        for size in sizes_to_try:
            try:
                messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": local_video_path,
                            "max_pixels": size,
                            "fps": fps,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }]

                # Preparation for inference
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to("cuda")

                # Inference
                generated_ids = model.generate(**inputs, max_new_tokens=128)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                # print(output_text)
                if isinstance(output_text, list):
                    output_text = output_text[0]
                logger.info(f"Size: {size}, Output text {output_text}.")
                all_results.append({"video_path": video_path, "question": question, "options": options, "gt_answer": d["answer"], "response": output_text, "id": d["id"], "size": size})
                break
            except Exception as e:
                logger.error(f"Failed to eval on {video_path}. {e}. Skipping...")
                if size == sizes_to_try[-1]:
                    all_results.append({"video_path": video_path, "question": question, "options": options, "gt_answer": d["answer"], "response": "-1", "id": d["id"], "error": str(e)})
                else:
                    logger.info(f"Retrying with next smaller size...")

    write_jsonl(all_results, out_fpath)


def compute_flops(
    benchmark_jsonl_fpath: str,
    out_fpath: str,
    template_prompt: str=MC,
    n_sample_frames: int=1,
):
    # let's measure vision component flops
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VisionTransformerPretrainedModel
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct", 
        attn_implementation="eager",
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    # vision_model = model.visual
    # inputs_vision = {
    #     "hidden_states": torch.randn(720, 1176).to("cuda"),
    #     "grid_thw": torch.tensor([[1, 20, 36]]).to("cuda")
    # }

    model = load_model(attn_implementation="eager")
    data = read_jsonl(benchmark_jsonl_fpath)
    all_results = []

    if os.path.exists(out_fpath):
        logger.info(f"Output file exists. Returning...")
        return

    sizes_to_try = [(28 * i) * (28 * i) for i in range(12, 7, -1)]
    
    for idx, d in enumerate(tqdm(data)):
        video_path = d["video_path"]
        question = d["question"]
        options = d["options"]
        local_video_path = f"/tmp/{Path(video_path).name}"

        try:
            wget.download(video_path, out=local_video_path, bar=None)
        except Exception as e:
            logger.error(f"Failed to process {video_path}. {e}. Skipping...") 

        prompt = template_prompt.format(options=options, question=question)

        video = cv2.VideoCapture(local_video_path)
        question = d["question"]

        # video.release()
        pil_frames = []
        frame_count = 0

        duration = video.get(cv2.CAP_PROP_POS_MSEC)
        fps = video.get(cv2.CAP_PROP_FPS)

        while True:
            ret, frame = video.read()
            if not ret:
                break
            # Check if the current frame is to be sampled
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

        # compute fps for the sampled frames
        sampled_fps = fps * len(pil_frames) / frame_count
        
        # write pil frames to files
        for i, pil_frame in enumerate(pil_frames):
            pil_frame.save(f"/tmp/{Path(video_path).stem}_{i}.jpg")

        for size in sizes_to_try:
            try:
                messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": [f"/tmp/{Path(video_path).stem}_{i}.jpg" for i in range(len(pil_frames))],
                            "fps": sampled_fps,
                            "max_pixels": size,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }]
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                logger.info(f"input shape: {inputs.input_ids.shape}, size: {size}")
                inputs = inputs.to("cuda")

                # Inference
                # generated_ids = model.generate(**inputs, max_new_tokens=512)
                inputs_tuple = (
                    inputs["input_ids"],
                    inputs["attention_mask"],
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    inputs["pixel_values_videos"],
                    None,
                    inputs["video_grid_thw"],
                )
                with torch.no_grad():
                    flop_analysis = FlopCountAnalysis(model, inputs=inputs_tuple)
                    flops = flop_analysis.total()
                    flops = flops / 1e9
                    logger.info(f"FLOPS: {flops} GFLOPS")
                
                all_results.append({"video_path": video_path, "question": question, "options": options, "gt_answer": d["answer"], "flops": flops, "id": d["id"], "size": size, "n_sample_frames": n_sample_frames})
                break
            except Exception as e:
                logger.error(f"Failed to process {video_path}. {e}. Skipping...")
                if size == sizes_to_try[-1]:
                    all_results.append({"video_path": video_path, "question": question, "options": options, "gt_answer": d["answer"], "flops": 0, "id": d["id"], "error": str(e)})
                else:
                    logger.info(f"Retrying with next smaller size...")
    
    # compute the average flops for non-error cases
    avg_flops = np.mean([r["flops"] for r in all_results if "error" not in r])
    logger.info(f"Average FLOPS: {avg_flops} GFLOPS")
        
    write_jsonl(all_results, out_fpath)    
    
    # # inputs_tuple = (inputs["hidden_states"], inputs["grid_thw"])
    # # inputs_tuple = (inputs["input_ids"], inputs["attention_mask"], inputs["pixel_values_videos"], inputs["grid_thw"])
    # # keep all the keys in the inputs and preserve the order
    # # output = vision_model(**inputs)
    # # output = model(**inputs)
    # # flop_analysis = FlopCountAnalysis(vision_model, inputs=inputs_tuple)
    # flop_analysis = FlopCountAnalysis(model, inputs=inputs_tuple)
    # flops = flop_analysis.total()
    # flops = flops / 1e9
    # logger.info(f"FLOPS: {flops} GFLOPS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_jsonl_fpath", type=str, help="benchmark jsonl file path")
    parser.add_argument("--template_prompt", type=str, help="template prompt", choices=["MC", "MC_WITH_DESCRIPTION", "MC_WITH_DESCRIPTION_COT", "MC_COT"])
    parser.add_argument("--out_fpath", type=str, help="output jsonl file path")
    parser.add_argument("--fps", type=float, default=None, help="fps")
    parser.add_argument("--n_sample_frames", type=int, default=None, help="n_sample_frames")
    parser.add_argument("--fn", type=str, help="function name", choices=["video_eval_fps", "video_eval_nframes", "compute_flops"])

    args = parser.parse_args()

    if args.fn == "video_eval_fps":
        with_description = False
        if args.template_prompt in ["MC_WITH_DESCRIPTION" or "MC_WITH_DESCRIPTION_COT"]:
            with_description = True
        elif args.template_prompt in ["MC" or "MC_COT"]:
            with_description = False

        video_eval_fps(
            # benchmark_jsonl_fpath="/net/nfs.cirrascale/prior/reza/action_atlas/final_benchmark_v3.jsonl",
            benchmark_jsonl_fpath=args.benchmark_jsonl_fpath,
            with_description=with_description,
            template_prompt=eval(args.template_prompt),
            # out_fpath="/net/nfs.cirrascale/prior/reza/action_atlas/final_benchmark_v3_output.jsonl"
            out_fpath=args.out_fpath,
            fps=args.fps,
        )
    elif args.fn == "video_eval_nframes":
        with_description = False
        if args.template_prompt in ["MC_WITH_DESCRIPTION" or "MC_WITH_DESCRIPTION_COT"]:
            with_description = True
        elif args.template_prompt in ["MC" or "MC_COT"]:
            with_description = False

        video_eval_nframes(
            benchmark_jsonl_fpath=args.benchmark_jsonl_fpath,
            template_prompt=eval(args.template_prompt),
            out_fpath=args.out_fpath,
            with_description=with_description,
            n_sample_frames=args.n_sample_frames,
        )
    elif args.fn == "compute_flops":
        compute_flops(
            benchmark_jsonl_fpath=args.benchmark_jsonl_fpath,
            out_fpath=args.out_fpath,
            template_prompt=MC,
            n_sample_frames=args.n_sample_frames,
        )
    else:
        raise ValueError("Invalid function name.")
    
    if args.fn in ["video_eval_fps", "video_eval_nframes"]:
        if args.template_prompt in ["MC", "MC_WITH_DESCRIPTION"]:
            compute_final_accuracy(args.out_fpath, verbose=True, conf_interval=True)
        elif args.template_prompt in ["MC_WITH_DESCRIPTION_COT", "MC_COT"]:
            compute_final_accuracy_cot(args.out_fpath, verbose=True, conf_interval=True)