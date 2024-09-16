import os
import sys
import time
from functools import partial
from typing import List, Union, Callable, Dict
from multiprocessing import Pool
from pathlib import Path

import openai
from openai import AzureOpenAI, RateLimitError, APIError, Timeout
import backoff
from tqdm import tqdm
from loguru import logger

from action_atlas.utils import write_jsonl


assert os.getenv("AZURE_OPENAI_ENDPOINT") is not None, "AZURE_OPENAI_ENDPOINT environment variable is not set."
assert os.getenv("AZURE_OPENAI_KEY") is not None, "AZURE_OPENAI_KEY environment variable is not set."


class MultiProcessCaller:
    """Call API in multiprocess"""
    @staticmethod
    def call_multi_process(fn: Callable, data: List, num_processes: int):
        with Pool(num_processes) as pool:
            pbar = tqdm(total=len(data))
            for result in pool.imap_unordered(fn, data):
                if result is not None:
                    pbar.update()
                    yield result


class AzureOpenAICaller:
    def __init__(self, max_time=60) -> None:
        self.client = AzureOpenAI(
            api_version="2024-04-01-preview",
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        self.max_time = max_time
 
    def robust_api_call(self, fn):
        return backoff.on_exception(
            backoff.expo,
            (APIError, Timeout, RateLimitError),
            max_time=self.max_time
        )(fn)()

    def complete_chat(
        self,
        messages,
        model='video-llm-gpt4v',
        max_tokens=1000,
        num_log_probs=None,
        n=1,
        top_p=1.0,
        temperature=0.3,
        stop=None,
        echo=True,
        frequency_penalty=0.,
        presence_penalty=0.,
    ):
        def _complete_chat():
            for _ in range(100):
                try:
                    completion = self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        max_tokens=max_tokens,
                    )
                    return completion
                except Exception as e:
                    if hasattr(e, 'code') and e.code == "content_filter":
                        raise e
                    logger.error(f"Error: {sys.exc_info()}. j")
                    time.sleep(10)
            return None

        return self.robust_api_call(_complete_chat)


def robust_api_call(fn):
    max_time = 60
    return backoff.on_exception(
        backoff.expo,
        (APIError, Timeout, RateLimitError),
        max_time=max_time
    )(fn)()


def complete_chat_single_process(
    messages,
    model='video-llm-gpt4v',  # either video-llm-gpt4v or video-llm-gpt4-1106-preview
    max_tokens=1000,
    num_log_probs=None,
    n=1,
    top_p=1.0,
    temperature=0.3,
    stop=None,
    echo=True,
    frequency_penalty=0.,
    presence_penalty=0.,
):
    client = AzureOpenAI(
        api_version="2024-04-01-preview",
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    messages, metadata = messages

    def _complete_chat():
        for _ in range(100):
            try:
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                )
                return completion, metadata
            except Exception as e:
                if hasattr(e, 'code') and e.code == "content_filter":
                    logger.warning(f"Content filter error: {e}. Returning None...")
                    return None, metadata

                logger.error('Error:', sys.exc_info())
                time.sleep(10)

        return None, metadata

    return robust_api_call(_complete_chat)


def complete_chat_multi_process(
    num_processes: int,
    all_messages: List,
    model='video-llm-gpt4v',
    max_tokens=1000,
    num_log_probs=None,
    n=1,
    top_p=1.0,
    temperature=0.3,
    stop=None, 
    echo=True,
    frequency_penalty=0., 
    presence_penalty=0.,
    **kwargs
):
    """Process chat completions in multiple processes."""
    fn = partial(
        complete_chat_single_process,
        model=model,
        max_tokens=max_tokens,
        num_log_probs=num_log_probs,
        n=n,
        top_p=top_p,
        temperature=temperature,
        stop=stop,
        echo=echo,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
    )
    results_it = MultiProcessCaller.call_multi_process(
        fn, all_messages,
        num_processes=num_processes
    )

    out_fprefix = kwargs.get("out_fprefix", None)

    if out_fprefix is None:
        return results_it

    assert "save_every" in kwargs, "save_every must be provided when out_fprefix is provided."
    assert "response_save_key" in kwargs, "response_save_key must be provided when out_fprefix is provided."

    save_every = kwargs.get("save_every", 1000)
    response_save_key = kwargs.get("response_save_key", "response")

    os.makedirs(out_fprefix, exist_ok=True)
    proc_results = []
    counter = 0

    for res in results_it:
        res, meta = res
        if res is not None:
            meta.update({response_save_key: res.choices[0].message.content.strip().splitlines()})
        else:
            meta.update({response_save_key: []})

        proc_results.append(meta)

        counter += 1

        if len(proc_results) % save_every == 0:
            write_jsonl(proc_results, os.path.join(out_fprefix, f"{str(counter).zfill(5)}_{str(len(all_messages)).zfill(5)}.jsonl"))
            proc_results = []

    if len(proc_results) > 0:
        write_jsonl(proc_results, os.path.join(out_fprefix, f"{str(counter).zfill(5)}_{str(len(all_messages)).zfill(5)}.jsonl"))


if __name__ == "__main__":
    # caller = AzureOpenAICaller()
    oai = AzureOpenAICaller()
    res = oai.complete_chat(
        messages=[{
            "role": "user",
            "content": "How you doin?",
            },
        ])
    print(res.choices[0].message.content)