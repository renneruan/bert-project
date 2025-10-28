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

    # Mantemos apenas coluna de Texto
    wikipedia = wikipedia.remove_columns(
        [col for col in wikipedia.column_names if col != "text"]
    )

    return wikipedia


def load_brwac(cached_data_folder: str):
    brwac = load_dataset(
        "dominguesm/brwac",
        # data_dir="dataset-brwac",
        split="train",
        num_proc=cpu_count(),
        cache_dir=cached_data_folder,
        trust_remote_code=True,
    )

    brwac = brwac.remove_columns(
        [col for col in brwac.column_names if col != "text"]
    )

    cleaned_brwac = brwac.map(
        paragraph_to_document,
        batched=True,
        remove_columns=["text"],
        num_proc=cpu_count(),
    )

    return cleaned_brwac


def load_all_datasets(cached_data_folder: str):
    logger.info("Loading all datasets")

    wikipedia = load_wikipedia_pages(cached_data_folder)
    brwac = load_brwac(cached_data_folder)

    raw_datasets = concatenate_datasets([wikipedia, brwac])

    logger.info("Finished loading all datasets")

    return raw_datasets
