import os
import logging
import nltk

from multiprocessing import cpu_count

from src.brazilian_modernbert.utils.text_helper import get_text_metadata

nltk.download("stopwords")
stopwords = nltk.corpus.stopwords.words("portuguese")

logger = logging.getLogger(__name__)


def clean_for_first_phase(dataset):
    cleaned_dataset = dataset.filter(
        lambda example: example["num_words"] >= 20
        and example["num_words"] <= 512
        and example["stopwords"] >= 1
        and example["average"] >= 2
        and example["average"] <= 15
    )

    cleaned_dataset = cleaned_dataset.remove_columns(
        [col for col in cleaned_dataset.column_names if col != "text"]
    )  # only keep the 'text' column
    return cleaned_dataset


def preprocess_concatenated_dataset(data_path, dataset):
    logger.info("Preprocessing concatenated dataset")

    preprocessed_dataset = dataset.map(
        get_text_metadata,
        batched=True,
        remove_columns=["text"],
        num_proc=cpu_count(),
    )

    preprocessed_dataset = preprocessed_dataset.rename_column(
        "paragraphs", "text"
    )

    logger.info("Cleaning for first training phase")
    cleaned_for_fist_phase = clean_for_first_phase(preprocessed_dataset)

    logger.info("Splitting dataset")
    split_dataset = cleaned_for_fist_phase.train_test_split(
        test_size=0.1, shuffle=True, seed=42
    )

    split_save_path = os.path.join(data_path, "split_datasets")
    split_dataset.save_to_disk(split_save_path)
    logger.info("Splitted dataset saved on %s", split_save_path)

    return split_dataset
