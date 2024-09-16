import os
import concurrent
from typing import List
from ssl import create_default_context
import json
from pathlib import Path
from tqdm import tqdm
import math
from collections import defaultdict
from typing import List, Dict, Any

from loguru import logger
from elasticsearch import Elasticsearch

from action_atlas.utils import (
    sanitize_file_name,
    read_jsonl,
    download_gcs_blob,
    write_jsonl,
)
from action_atlas.azure_openai import complete_chat_multi_process
from action_atlas.prompts import GPT4_FILTER_PARTIAL_MATCH


DOMAINS_NAME = [el["name"] for el in read_jsonl("data/domains.jsonl")]


class ElasticSearchHandler:
    def __init__(self, es_url="http://localhost:9200", es_user='elastic', es_password=''):
        self.es = Elasticsearch(
            hosts=[es_url],
            basic_auth=(es_user, es_password),
            verify_certs=False,
        )

    def create_index(self, index_name: str, settings=None, mappings=None):
        if not self.es.indices.exists(index=index_name):
            try:
                self.es.indices.create(
                    index=index_name,
                    ignore=400,
                    body={
                        "settings": settings or {
                            "number_of_shards": 1,
                            "number_of_replicas": 0,
                            "similarity": {
                                "default": {
                                    "type": "BM25"
                                }
                        }
                        },
                        "mappings": mappings or {
                            "properties": {
                                "chapters": {
                                    "properties": {
                                        "start_time": {"type": "double"},
                                        "end_time": {"type": "double"}
                                    }
                                }
                            }
                        }
                    }
                )
                logger.info(f"Index '{index_name}' created successfully.")
            except Exception as e:
                logger.error(f"Error creating index: {e}")
        else:
            logger.info("Index already exists")

    def index_document(self, index_name, document, verbose=False, max_attempts=3):
        for attempt in range(max_attempts):
            try:
                self.es.index(index=index_name, document=document)
                if verbose:
                    logger.info(f"Document {document["id"]} indexed successfully.")
                return 0
            except Exception as e:
                if verbose:
                    logger.info(f"Retry indexing video {document["id"]}")

        logger.error(f"Failed to index document {document["id"]}: {e}")
        return -1

    def refresh_index(self):
        self.es.indices.refresh(index=self.index_name)
        logger.info("Index refreshed successfully.")

    def search_index(
        self,
        index_name: str,
        query: Dict[str, Any],
        max_attempts: int = 3,
        verbose: bool = False
    ) -> Dict[str, Any]:
        for attempt in range(max_attempts):
            try:
                response = self.es.search(index=index_name, body=query)
                return dict(response)
            except Exception as e:
                if verbose:
                    logger.info(f"Attempt {attempt} to search index '{index_name}' for query {query} failed: {e}")

        logger.error(f"Error searching index {index_name} for query {query}: {e}")
        return {}


    def get_mapping(self):
        mapping = self.es.indices.get_mapping(index=self.index_name)
        logger.info("Index mapping:", mapping)
        return mapping
  
    def delete_index(self, index_name: str):
        """Deletes the specified index and all of its documents."""
        try:
            response = self.es.indices.delete(index=index_name)
            logger.info(f"Index '{index_name}' deleted successfully.")
        except Exception as e:
            logger.error(f"Error deleting index '{index_name}': {e}")
    
    def delete_index_documents(self, index_name: str):
        """Deletes all the documents in the specified index. Does not delete the index itself."""
        query = {
            "query": {
                "match_all": {}
            }
        }
        try:
            response = self.es.delete_by_query(index=index_name, body=query)
        except Exception as e:
            logger.error(f"Error deleting documents from index '{index_name}': {e}")

        logger.info(f"Deleted {response['deleted']} documents from index '{index_name}'.")

    def count_index_documents(self, index_name: str):
        """Returns the number of documents in the specified index."""
        response = self.es.count(index=index_name)
        return response["count"]

    @staticmethod
    def parse_documents(root_dir: str):
        docs = Path(root_dir).glob("*")
        video_id_to_metadata = defaultdict(dict)
        video_ids_with_corrupted_files = set()

        for doc in docs:
            video_id = doc.name.split(".")[0]
            if doc.as_posix().endswith(".description"):
                try:
                    content = doc.read_text()
                    if not video_id in video_ids_with_corrupted_files:
                        video_id_to_metadata[video_id]["description"] = content
                except Exception as e:
                    logger.error(f"Error reading file '{doc}': {e}. Skipping...")
                    video_ids_with_corrupted_files.add(video_id)

            elif doc.as_posix().endswith(".info.json"):
                try:
                    if not video_id in video_ids_with_corrupted_files:
                        blacklist_keys = [
                            "formats", "requested_formats", "requested_subtitles", "requested_entries",
                            "automatic_captions", "thumbnails", "requested_downloads", "_format_sort_fields",
                            "http_headers", "subtitles", "description"
                        ]
                        content = json.load(open(doc))
                        # finding domain name
                        playlist_title = content.get("playlist_title")
                        domain = None
                        for domain_name in DOMAINS_NAME:
                            if playlist_title.startswith(domain_name):
                                assert domain is None, f"Multiple domains found in playlist title: {playlist_title}"
                                domain = domain_name
                        action = playlist_title.replace(domain, "").strip()

                        content["domain"] = domain
                        content["action"] = action

                        video_id_to_metadata[video_id].update({k: v for k, v in content.items() if k not in blacklist_keys})
                        if video_id_to_metadata[video_id].get("subtitles") is None:
                            video_id_to_metadata[video_id]["subtitles"] = ""
                        if video_id_to_metadata[video_id].get("description") is None:
                            video_id_to_metadata[video_id]["description"] = ""
                except Exception as e:
                    logger.error(f"Error reading file '{doc}': {e}. Skipping...")
                    video_ids_with_corrupted_files.add(video_id)

            elif doc.as_posix().endswith(".en.vtt"):
                try:
                    if not video_id in video_ids_with_corrupted_files:
                        content = doc.read_text()
                        video_id_to_metadata[video_id]["subtitles"] = content
                except Exception as e:
                    logger.error(f"Error reading file '{doc}': {e}. Skipping...")
                    video_ids_with_corrupted_files.add(video_id)

        return video_id_to_metadata

    @staticmethod
    def parse_documents_from_merged_metadata_files(merged_info_json_fpath: str):
        """Indexes metadata from merged metadata files."""
        video_id_to_metadata = defaultdict(dict)
        try:
            data = json.load(open(merged_info_json_fpath))
            for video_id, content in data.items():
                if video_id.endswith(".info"):
                    video_id = video_id[:-5]
                try:
                    blacklist_keys = [
                        "formats", "requested_formats", "requested_subtitles", "requested_entries",
                        "automatic_captions", "thumbnails", "requested_downloads", "_format_sort_fields",
                        "http_headers", "subtitles", "description"
                    ]
                    playlist_title = content.get("playlist_title")
                    domain = None
                    for domain_name in DOMAINS_NAME:
                        if playlist_title.startswith(domain_name):
                            assert domain is None, f"Multiple domains found in playlist title: {playlist_title}"
                            domain = domain_name
                    action = playlist_title.replace(domain, "").strip()

                    content["domain"] = domain
                    content["action"] = action

                    video_id_to_metadata[video_id].update({k: v for k, v in content.items() if k not in blacklist_keys})
                except Exception as e:
                    logger.error(f"Error reading file '{video_id}': {e}. Skipping...")
        except Exception as e:
            logger.error(f"Error reading file '{merged_info_json_fpath}': {e}. Skipping...")

        return video_id_to_metadata


def index_documents_multithread(
    es_handler: ElasticSearchHandler,
    index_name: str,
    root_dir: str,
    max_workers: int = 5,
    verbose: bool = False
) -> None:
    """
    Index documents into Elasticsearch using multithreading.

    Args:
        es_handler (ElasticSearchHandler): The Elasticsearch handler instance.
        index_name (str): The name of the Elasticsearch index.
        root_dir (str): The root directory containing documents to be indexed.
        max_workers (int): The maximum number of worker threads.
        verbose (bool): Flag to enable verbose logging.
    """
    def _index_document(file_path):
        with open(file_path, 'r') as file:
            file_content = json.load(file)

        try:
            es_handler.index_document(file_content)
        except Exception as e:
            logger.error(f"Error indexing '{file_path}': {e}")

    es_uploader = ElasticSearchHandler(index_name=index_name)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for root, _, files in os.walk(root_dir):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                futures.append(executor.submit(_index_document, file_path, es_uploader))

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error in future: {e}")


def exact_title_match(
    out_fpath: str,
    size: int=100,
    save_every: int=500,
):
    """Videos with phrase match in the title of the video and more than 1000 view counts."""
    domain_action_names_to_domain_and_action_map = json.load(open("../data/query_to_domain_and_action_mapping.json"))
    indexed_actions = os.listdir("/mnt/elasticsearch/actions/")
    indexed_actions = [el.replace("_merged_info.json", "") for el in indexed_actions]

    all_res = {}
    counter = 0
    for idx, domain_action in enumerate(tqdm(indexed_actions)):
        try:
            domain, action = domain_action_names_to_domain_and_action_map[domain_action]
        except:
            logger.warning(f"Domain action not found: {domain_action}")
            continue

        query = {
            "size": size,
            "query": {
                "bool": {
                    "must": [
                        {
                            "range": {
                                "view_count": {
                                    "gte": 1000
                                }
                            }
                        },
                        {
                            "match_phrase": {
                                "domain": domain
                            }
                        },
                        {
                            "match_phrase": {
                                "title": action
                            }
                        }
                    ]
                }
            }
        }
        res = es_handler.search_index(index_name="video_benchmark", query=query)
        all_res[domain_action] = res
        counter += 1
        if counter % save_every == 0:
            with open(out_fpath.replace(".json", f"_shard_{counter//save_every}.json"), "w") as f:
                json.dump(all_res, f, indent=2)
            all_res = {}

    if len(all_res) > 0:
        with open(out_fpath.replace(".json", f"_shard_{math.ceil(counter / save_every)}.json"), "w") as f:
            json.dump(all_res, f, indent=2)


def partial_title_match(
    out_fpath: str,
    tier1_fpath: str,
    size: int=100,
):
    """These are the videos where the action name does not exactly appear in the title 
    but partially appears or is a synonym of the action name.
    These video titles would be further checked using GPT4 to see if the action is actually happening in the video
    or if some other action is actually happening.
    """
    domain_action_names_to_domain_and_action_map = json.load(open("../data/query_to_domain_and_action_mapping.json"))
    indexed_actions = os.listdir("/mnt/elasticsearch/actions_shard_1_4/")
    indexed_actions = [el.replace("_merged_info.json", "") for el in indexed_actions]
    tier1_res = json.load(open(tier1_fpath))
    all_res = {}

    for idx, domain_action in enumerate(tqdm(indexed_actions)):
        domain_action = sanitize_file_name(domain_action)
        try:
            domain, action = domain_action_names_to_domain_and_action_map[domain_action]
        except:
            logger.warning(f"Domain action not found in mapping: {domain_action}")
            continue

        query = {
            "size": size,
            "query": {
                "bool": {
                    "must": [
                        {
                            "range": {
                                "view_count": {
                                    "gte": 1000
                                }
                            }
                        },
                        {
                            "match_phrase": {
                                "domain": domain
                            }
                        },
                        {
                            "match": {
                                "title": action
                            }
                        }
                    ]
                }
            }
        }
        res = es_handler.search_index(index_name="video_benchmark", query=query)

        # filter out tier1 results
        tier1_res_ids = set([el['_source']['id'] for el in tier1_res[domain_action]['hits']['hits']])
        res['hits']['hits'] = [el for el in res['hits']['hits'] if el['_source']['id'] not in tier1_res_ids]
        res['hits']['total']['value'] = len(res['hits']['hits'])
        all_res[domain_action] = res

    with open(out_fpath, "w") as f:
        json.dump(all_res, f, indent=2)


def partial_title_match_with_gpt4_filtering():
    """
    Tier3 video title do not contain the exact action name but a very similar phrase that might be a synonym of the action name or a different action.
    Filter tier3 videos using GPT4 to see if the action is actually happening in the video or if some other action is actually happening.
    """
    tier2_data = json.load(open("../data/tier1_1200_actions.json"))
    domain_action_to_domain_and_action = json.load(open("../data/query_to_domain_and_action_mapping.json"))

    prompts = []
    prompts = []
 
    meta = []
    prompts = []
    for domain_action, v in tier2_data.items():
        domain, action = domain_action_to_domain_and_action[domain_action]
        titles = [el["_source"]["title"] for el in v["hits"]["hits"]]
        prompts.extend([GPT4_FILTER_PARTIAL_MATCH.format(title=title, sport=domain, action=action) for title in titles])
        meta.extend([el["_source"] for el in v["hits"]["hits"]])

    all_messages = [[{"role": "user", "content": p}] for p in prompts]
    all_messages = [(message, m) for message, m in zip(all_messages, meta)]
    ###
    complete_chat_multi_process(
        all_messages=all_messages,
        num_processes=8,
        out_fprefix="/mnt/elasticsearch/partial_title_match_with_gpt4_filtering",
        save_every=1000,
        response_save_key="response",
    )


def search_for_single_action(action: str, domain: str):
    # action = "Toe Poke"
    # action = "Spin Layup"
    # action = "Knuckleball"
    query = {
        "size": 100,
        "query": {
            "bool": {
                "must": [
                    {
                        "range": {
                            "view_count": {
                                "gte": 1000
                            }
                        }
                    },
                    {
                        "match_phrase": {
                            "domain": domain,
                        }
                    },
                    {
                        "match": {
                            "title": action
                        }
                    }
                ]
            }
        }
    }
    res = es_handler.search_index(index_name="video_benchmark", query=query)
    print("found documents:", res['hits']['total']['value'])
    with open(f"../data/{action}.json", "w") as f:
        json.dump(res, f, indent=2)


def parse_and_index_documents(
    index_name: str,
    action_names: List[str],
    actions_root_dir: str = "/mnt/elasticsearch/actions_v2",
    delete_existing_docs: bool = True,
) -> None:
    """
    Parse and index documents into Elasticsearch.

    Args:
        index_name (str): The name of the Elasticsearch index.
        action_names (List[str]): List of action names to be indexed.
        actions_root_dir (str): Root directory containing action documents.
        delete_existing_docs (bool): Flag to delete existing documents in the index.
    """
    es_handler = ElasticSearchHandler()
    
    # Create the index if it does not exist
    es_handler.create_index(index_name=index_name)
    
    # Optionally delete existing documents in the index
    if delete_existing_docs:
        es_handler.delete_index_documents(index_name=index_name)

    for action_dir in tqdm(action_names, desc="Indexing documents"):
        docs = es_handler.parse_documents(f"{actions_root_dir}/{action_dir}")
        for video_id, meta in docs.items():
            es_handler.index_document(index_name="video_benchmark", document=meta)


def parse_and_index_documents_multithread(
    index_name: str,
    action_names: List[str],
    actions_root_dir:str="/mnt/elasticsearch/actions_v2",
    delete_existing_docs=True,
    max_workers:int=16,
):
    def _parse_index_single_action(action_dir):
        docs = es_handler.parse_documents(f"{actions_root_dir}/{action_dir}")
        for video_id, meta in docs.items():
            es_handler.index_document(index_name="video_benchmark", document=meta)

        with open(f"../data/indexed_actions.txt", "a") as f:
            f.write(f"{action_dir}\n")

    es_handler = ElasticSearchHandler()
    es_handler.create_index(index_name=index_name)
    if delete_existing_docs:
        es_handler.delete_index_documents(index_name=index_name)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for action_dir in tqdm(action_names):
            futures.append(executor.submit(_parse_index_single_action, action_dir))
 
        # Wait for all threads to complet
        for future in tqdm(futures, total=len(action_names), desc="Parsing and indexing docs"):
            future.result()


def parse_and_index_documents_from_merged_metadata_files(
    index_name: str,
    actions_root_dir:str="/mnt/elasticsearch/actions_v2",
):
    es_handler = ElasticSearchHandler()
    es_handler.create_index(index_name=index_name)

    action_merged_files = os.listdir(actions_root_dir)

    # with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # futures = []
    for action_merged_file in tqdm(action_merged_files):
        action_merged_file = f"{actions_root_dir}/{action_merged_file}"
        docs = es_handler.parse_documents_from_merged_metadata_files(action_merged_file)
        for video_id, meta in docs.items():
            es_handler.index_document(index_name="video_benchmark", document=meta)
            logger.info(f"Indexed video: {video_id}")

        with open(f"../data/indexed_actions.txt", "a") as f:
            f.write(f"{action_merged_file}\n")


def parse_and_index_documents_multithread_from_merged_metadata_files(
    index_name: str,
    actions_root_dir:str="/mnt/elasticsearch/actions_v2",
    delete_existing_docs=True,
    max_workers:int=16,
):
    def _parse_index_merged_file(merged_file):
        docs = es_handler.parse_documents_from_merged_metadata_files(merged_file)
        for video_id, meta in docs.items():
            es_handler.index_document(index_name="video_benchmark", document=meta)

        with open(f"../data/indexed_actions.txt", "a") as f:
            f.write(f"{action_dir}\n")

    es_handler = ElasticSearchHandler()
    es_handler.delete_index(index_name=index_name)
    es_handler.create_index(index_name=index_name)

    action_merged_files = os.listdir(actions_root_dir)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for action_merged_file in action_merged_files:
            action_dir = f"{actions_root_dir}/{action_merged_file}"
            futures.append(executor.submit(_parse_index_merged_file, action_dir))
 
        # Wait for all threads to complete
        for future in tqdm(futures, total=len(action_merged_files), desc="Parsing and indexing docs"):
            future.result()


def select_actions_for_transcription(
    tier1_fpath: str,
):
    """
    For now only selecting tier1 and tier2 actions for transcription.
    tier3 titles should be further processed with GPT4 to see if the video is about the
    action or it might contain some action that does not exist in our list of actions.
    """
    tier1_data = json.load(open(tier1_fpath))

    physical_actions = json.load(open("../data/physical_actions.json"))

    counts = []
    for domain_action, v in tier1_data.items():
        if domain_action in physical_actions:
            count = len(v['hits']['hits'])
            counts.append((domain_action, count))

    counts = sorted(counts, key=lambda x: x[1], reverse=True)

    ## top100 actions here are good. let's download 100 videos for each of them and see how we can localize them better.
    count_sum = 0
    metadata = []
    num_actions = 0
    for idx, c in enumerate(counts):
        if c[1] > 0:
            logger.info(f"{idx+1}. {c[0]}: {c[1]}")
            num_actions += 1
            # search_results_1 = json.load(open(f"../search_results_priority_1/{c[0]}.json"))
            # search_results_1 = json.load(open(f"{tier1_dir}/{c[0]}.json"))
            # search_results_2 = json.load(open(f"{tier2_dir}/{c[0]}.json"))
            # ids = [el["_source"]["id"] for el in search_results_1["hits"]["hits"]] + \
                # [el["_source"]["id"] for el in search_results_2["hits"]["hits"]]
            search_results_1 = tier1_data[c[0]]
            ids1 = [el["_source"]["id"] for el in search_results_1["hits"]["hits"]]
            meta1 = [el["_source"] for el in search_results_1["hits"]["hits"]]
            for id_, meta in zip(ids1, meta1):
                assert id_ == meta["id"]
                m = {"YoutubeID": id_, "domain_action": c[0], "priority": 1}
                m.update(meta)
                metadata.append(m)

            # ids2 = [el["_source"]["id"] for el in search_results_2["hits"]["hits"]]
            # meta2 = [el["_source"] for el in search_results_2["hits"]["hits"]]
            # for id_, meta in zip(ids2, meta2):
            #     assert id_ == meta["id"]
            #     m = {"YoutubeID": id_, "domain_action": c[0], "priority": 2}
            #     m.update(meta)
            #     metadata.append(m)

        count_sum += c[1]

    logger.info(f"Total number of actions: {num_actions}")
    write_jsonl(metadata, f"../data/top{num_actions}_elasticsearch_actions_metadata.jsonl")


if __name__ == "__main__":
    es_handler = ElasticSearchHandler()
    es_handler.create_index(index_name="video_benchmark")

    partial_title_match(
        out_fpath="/mnt/elasticsearch/batch3_shard1_tier2.json",
        tier1_fpath="/mnt/elasticsearch/batch3_shard1_tier1.json",
        size=100,
    )