import logging
from multiprocessing import cpu_count

import transformers
from datasets import load_dataset, concatenate_datasets

from src.brazilian_modernbert.utils.text_helper import paragraph_to_document

print(transformers.__version__)


logger = logging.getLogger(__name__)


def load_wikipedia_pages(cached_data_folder: str):
    wikipedia = load_dataset(
        "wikimedia/wikipedia",
        "20231101.pt",
        split="train",
        num_proc=cpu_count(),
        cache_dir=cached_data_folder,
    )

    return wikipedia


def load_brwac(cached_data_folder: str):
    brwac = load_dataset(
        "dominguesm/brwac",
        # data_dir="dataset-brwac",
        split="train",
        num_proc=cpu_count(),
        cache_dir=cached_data_folder,
        # trust_remote_code=True,
    )

    cleaned_brwac = brwac.map(
        paragraph_to_document,
        batched=True,
        remove_columns=["text"],
        num_proc=cpu_count(),
    )

    return cleaned_brwac

def load_ccpt(cached_data_folder: str):
    ccpt = load_dataset(
        "ClassiCC-Corpus/ClassiCC-PT",
        split="train",
        num_proc=4,
        cache_dir=cached_data_folder
    )

    return ccpt


def load_all_datasets(cached_data_folder: str):
    logger.info("Loading all datasets")

    # wikipedia = load_wikipedia_pages(cached_data_folder)
    # brwac = load_brwac(cached_data_folder)
    ccpt = load_ccpt(cached_data_folder)

    # wikipedia = wikipedia.select_columns(["text"])
    # brwac = brwac.select_columns(["text"])
    ccpt = ccpt.select_columns(["text"])

    # raw_datasets = concatenate_datasets([wikipedia, brwac])
    raw_datasets = ccpt
    
    logger.info("Finished loading all datasets")

    return raw_datasets
