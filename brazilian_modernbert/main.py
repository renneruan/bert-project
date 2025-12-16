import os
import logging

from src.brazilian_modernbert.configs import (
    CACHED_DATA_FOLDER,
    WORK_DIR,
    DATA_FOLDER,
    VOCABULARY_SIZE,
    CONTEXT_SIZE,
    LOAD_AND_PREPROCESS_DATASET,
    TRAIN_TOKENIZER,
)

import datasets
from datasets import load_from_disk

from src.brazilian_modernbert.logging import setup_logging

setup_logging()

logger = logging.getLogger(__name__)

from src.brazilian_modernbert.utils.health_check import health_check

from src.brazilian_modernbert.data_collect.load_datasets import (
    load_all_datasets,
)

from src.brazilian_modernbert.data_preprocessing.preprocess_datasets import (
    preprocess_concatenated_dataset,
)

from src.brazilian_modernbert.data_preprocessing.tokenizer import (
    create_and_train_tokenizer,
    evaluate_fertility,
    tokenize_dataset_with_padding,
)


def main():
    logger.info(f"HF Default Cache: {datasets.config.HF_DATASETS_CACHE}")
    logger.info(f"Custom Cache Target: {CACHED_DATA_FOLDER}")

    split_save_path = os.path.join(DATA_FOLDER, "split_datasets")

    if LOAD_AND_PREPROCESS_DATASET:
        logger.info(
            "LOAD_AND_PREPROCESS_DATASET is True. Running full preprocessing..."
        )
        raw_datasets = load_all_datasets(cached_data_folder=CACHED_DATA_FOLDER)
        splitted_dataset = preprocess_concatenated_dataset(
            DATA_FOLDER, raw_datasets
        )
    else:
        logger.info(
            f"LOAD_AND_PREPROCESS_DATASET is False. Loading from {split_save_path}..."
        )
        if not os.path.isdir(split_save_path):
            logger.error(
                f"Dataset not found at {split_save_path}. Set LOAD_AND_PREPROCESS_DATASET=True in configs to create it."
            )
            raise FileNotFoundError(f"Directory not found: {split_save_path}")

        logger.info(
            f"Loading pre-saved split dataset from {split_save_path}..."
        )
        splitted_dataset = load_from_disk(split_save_path)

    tokenizer = create_and_train_tokenizer(
        VOCABULARY_SIZE, CONTEXT_SIZE, splitted_dataset, TRAIN_TOKENIZER
    )

    # evaluate_fertility(tokenizer, splitted_dataset)

    tokenized_dataset = tokenize_dataset_with_padding(
        data_folder=DATA_FOLDER,
        tokenizer=tokenizer,
        vocabulary_size=VOCABULARY_SIZE,
        context_size=CONTEXT_SIZE,
        input_dataset=splitted_dataset["train"],
    )
    logger.info(tokenized_dataset)


if __name__ == "__main__":
    health_check()

    main()
