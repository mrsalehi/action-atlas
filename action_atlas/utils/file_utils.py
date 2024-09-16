from typing import Dict, Any, Generator, Union, List
import json
from pathlib import Path
import regex as re


def read_jsonl(path: Union[str, Path]) -> List[dict]:
    with open(path, 'r') as jsonl_file:
        return [json.loads(line) for line in jsonl_file]


def stream_jsonl(path: str) -> Generator[Dict[str, Any], None, None]:
    """Generator that streams jsonl file line by line"""
    with open(path, 'r') as jsonl_file:
        for line in jsonl_file:
            yield json.loads(line)

 
def write_jsonl(data, path):
    with open(path, 'w') as jsonl_file:
        for line in data:
            jsonl_file.write(json.dumps(line) + '\n')


def sanitize_file_name(input_string: str) -> str:
    """
    Sanitizes an input string to be used as a file name by replacing any sequence of 
    non-alphanumeric characters with a single underscore, and converting the string to lowercase.
    
    Args:
        input_string (str): The input string to be sanitized.
    
    Returns:
        str: The sanitized string suitable for use as a file name.
    """
    pattern = re.compile(r'[^a-zA-Z0-9]+')
    
    # Replace non-alphanumeric characters with underscores, strip leading/trailing underscores, and convert to lowercase
    sanitized_string = pattern.sub('_', input_string).strip('_').lower()
    
    return sanitized_string