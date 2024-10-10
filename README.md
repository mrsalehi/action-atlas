# ActionAtlas (NeurIPS 2024 D&B)

[[Website]](https://mrsalehi.github.io/action-atlas/)
[![arXiv](https://img.shields.io/badge/arXiv-2104.00001-red.svg)](https://arxiv.org/abs/2410.05774)

This is the official repository for the ActionAtlas benchmark. The benchmark evaluates large multimodal-models on videos of complex actions in specialized domains. This first version of the benchmark focuses on sports moves.



## Installation
You can install the package from source with pip or poetry:
```bash
pip install -e .
```

## Usage
1. Download the metadta either from this [google drive](https://drive.google.com/file/d/1ueh5gqYg0WqQ_CFxjxsjcn8rx9wwN9Gi/view?usp=drive_link) or from [HuggingFace](https://huggingface.co/datasets/action_atlas)

2. Each sample in the metadata contains a YouTube ID and the metadata of that video. Download the video from YouTube. Please take a look at `action_atlas/download_yt_videos.py` provided in this repo.

3. Extract ActionAtlas video segments from the original videos using `action_atlas/extract_segments.py`:
```bash
python action_atlas/extract_segments.py \
    --data_fpath /path/to/metadata.json \
    --yt_videos_dir /path/to/downloaded_yt_videos \
    --out_segments_dir /path/to/output_dir/for/segments \
    --max_workers 32
```

4. There is text on some of the videos that leak information about the action. We have already found polygons obfuscating the text using Google Cloud Vision API and provided them in the metadata. You can reused them to obfuscate text by running `action_atlas/obfuscate_text.py`:
```bash
python action_atlas/obfuscate_text.py  \
    obfuscate_text_in_videos_with_masks \
    --data_fpath /path/to/metadata.json \
    --video_segments_dir /path/to/extracted_segments \
    --out_dir /path/to/output_dir/for/final/segments/including/obfuscated \
    --max_workers 32
```
Note that after running the above command all videos in ActionAtlas will be stored in `out_dir`, including those with obfuscated text.

5. We have provided example script to evaluate both proprietary and open models. Please take a look at `action_atlas/eval_proprietary.py` and `action_atlas/eval_qwen2_vl.py`.


## Citation
If you use this dataset in your research, please cite the following paper:
```
@misc{salehi2024actionatlasvideoqabenchmarkdomainspecialized,
      title={ActionAtlas: A VideoQA Benchmark for Domain-specialized Action Recognition}, 
      author={Mohammadreza Salehi and Jae Sung Park and Tanush Yadav and Aditya Kusupati and Ranjay Krishna and Yejin Choi and Hannaneh Hajishirzi and Ali Farhadi},
      year={2024},
      eprint={2410.05774},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.05774}, 
}
```