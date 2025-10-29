import os
import logging

from src.brazilian_modernbert.configs import (
    CACHED_DATA_FOLDER,
    WORK_DIR,
    DATA_FOLDER,
    VOCABULARY_SIZE,
    CONTEXT_SIZE,
)
from src.brazilian_modernbert.logging import setup_logging
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
    tokenize_dataset_with_sequence_packing,
)

setup_logging()

logger = logging.getLogger(__name__)


def main():
    logging.info()

    raw_datasets = load_all_datasets(cached_data_folder=CACHED_DATA_FOLDER)

    splitted_dataset = preprocess_concatenated_dataset(raw_datasets)

    tokenizer = create_and_train_tokenizer(
        VOCABULARY_SIZE, CONTEXT_SIZE, splitted_dataset
    )

    evaluate_fertility(tokenizer, splitted_dataset)

    tokenized_dataset = tokenize_dataset_with_sequence_packing(
        data_folder=DATA_FOLDER,
        tokenizer=tokenizer,
        vocabulary_size=VOCABULARY_SIZE,
        context_size=CONTEXT_SIZE,
        input_dataset=splitted_dataset,
    )
    logger.info(tokenized_dataset)


if __name__ == "__main__":
    health_check()

    main()
