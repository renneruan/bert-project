import os
import logging
from datasets import load_from_disk

import numpy as np
from transformers import PreTrainedTokenizerFast
from tokenizers.normalizers import NFC, Lowercase, Replace
from tokenizers import normalizers, pre_tokenizers, Tokenizer

from tokenizers.pre_tokenizers import (
    Punctuation,
    Metaspace,
    WhitespaceSplit,
    Digits,
)
from tokenizers.processors import TemplateProcessing
from tokenizers.decoders import Metaspace as MetaspaceDecoder
from tokenizers.models import Unigram
from tokenizers.trainers import UnigramTrainer

from tqdm import tqdm

from multiprocessing import cpu_count

logger = logging.getLogger(__name__)

CUSTOM_SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


class TokenizerHandler:
    def __init__(self, vocabulary_size, context_size, input_dataset):

        self.custom_special_tokens = CUSTOM_SPECIAL_TOKENS
        self.vocabulary_size = vocabulary_size
        self.context_size = context_size
        self.input_dataset = input_dataset

        self._set_custom_normalizer()
        self._set_custom_pre_tokenizer()
        self._set_custom_post_processor()

        self._set_custom_tokenizer()

    def _set_custom_normalizer(self):
        # replace latex usage of "aspas"
        # NFC - Canonico (evita equivalencia) e composto, menos characteres
        # lowercase - unifica maiuscula e mi nusculo, que geralmente muda pouco (o positional embedding pode cuidar de comecar com maiuscula)

        # NFC, NFK
        # Ã A~

        self.custom_normalizer = normalizers.Sequence(
            [
                Replace("``", '"'),
                Replace("''", '"'),
                NFC(),
                Lowercase(),
            ]
        )

    def _set_custom_pre_tokenizer(self):
        self.custom_pre_tokenizer = pre_tokenizers.Sequence(
            [
                WhitespaceSplit(),
                Punctuation(),
                Digits(individual_digits=False),
                Metaspace(replacement="▁", prepend_scheme="always"),
            ]
        )

    def _set_custom_post_processor(self):
        self.custom_post_processor = TemplateProcessing(
            single=f"[CLS]:0 $A:0 [SEP]:0",
            pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", self.custom_special_tokens.index("[CLS]")),
                ("[SEP]", self.custom_special_tokens.index("[SEP]")),
            ],
        )

    def _set_custom_tokenizer(self):
        custom_tokenizer = Tokenizer(Unigram())

        custom_tokenizer.normalizer = self.custom_normalizer
        custom_tokenizer.pre_tokenizer = self.custom_pre_tokenizer
        custom_tokenizer.post_processor = self.custom_post_processor
        custom_tokenizer.decoder = MetaspaceDecoder()

        self.custom_tokenizer = custom_tokenizer

    def train_tokenizer(self):
        custom_trainer = UnigramTrainer(
            vocab_size=self.vocabulary_size,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=self.custom_special_tokens,
            unk_token="[UNK]",
        )

        # create a python generator to dynamically load the data, one batch at a time
        def batch_iterator(batch_size=128):  # 128 (cores)
            for i in tqdm(
                range(0, len(self.input_dataset["train"]), batch_size)
            ):
                yield self.input_dataset["train"][i : i + batch_size]["text"]

        self.custom_tokenizer.train_from_iterator(
            iterator=batch_iterator(), trainer=custom_trainer
        )

    def update_to_fast_tokenizer(self):
        self.fast_custom_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=self.custom_tokenizer,
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            bos_token="[CLS]",
            sep_token="[SEP]",
            eos_token="[SEP]",
            mask_token="[MASK]",
            padding_side="right",
        )

    def save_tokenizer(self):
        tokenizer_name = f"tokenizers/custom/{self.vocabulary_size:_}"
        self.fast_custom_tokenizer.save_pretrained(tokenizer_name)

        return self.fast_custom_tokenizer


def evaluate_fertility(tokenizer, dataset):
    logger.info("Calculating Tokenizer Fertility...")

    def calculate_per_doc(batch):
        encodings = tokenizer(
            batch["text"],
            add_special_tokens=False,
            truncation=False,
            padding=False,
        )
        generated_counts = [len(ids) for ids in encodings["input_ids"]]

        original_counts = []
        backend = tokenizer.backend_tokenizer

        for text in batch["text"]:
            normalized = backend.normalizer.normalize_str(text)
            pre_tokenized = backend.pre_tokenizer.pre_tokenize_str(normalized)
            original_counts.append(len(pre_tokenized))

        ratios = [
            gen / orig if orig > 0 else 0
            for gen, orig in zip(generated_counts, original_counts)
        ]

        return {
            "n_subwords": generated_counts,
            "n_words": original_counts,
            "fertility": ratios,
        }

    dataset_with_fertility = dataset["train"].map(
        calculate_per_doc,
        batched=True,
        batch_size=50000,
        num_proc=16,
        remove_columns=["text"],
        desc="Calculating Fertility",
    )

    logger.info("Aggregating statistics iteratively to save RAM...")

    total_fertility = 0
    max_fertility = 0
    count = len(dataset_with_fertility)

    for i in range(0, count, 100_000):
        chunk = dataset_with_fertility[i : i + 100_000]["fertility"]

        chunk_max = max(chunk)
        if chunk_max > max_fertility:
            max_fertility = chunk_max

        total_fertility += sum(chunk)

    avg_fertility = total_fertility / count

    logger.info(f"Fertility Report")
    logger.info(f"Average Fertility: {avg_fertility:.4f}")
    logger.info(f"Max Fertility:     {max_fertility:.4f}")

    return dataset_with_fertility


def analyze_token_distribution(
    dataset, tokenizer, threshold=1000, sample_ratio=0.01
):
    logger.info(f"Checking for documents with > {threshold} tokens...")

    total_rows = len(dataset)

    if 0 < sample_ratio < 1.0:
        step = int(1 / sample_ratio)
        indices = range(0, total_rows, step)
        sampled_dataset = dataset.select(indices)
        logger.info(
            f"Downsampling for analysis: Selected {len(sampled_dataset)} samples ({sample_ratio*100}%) from {total_rows} total."
        )
    else:
        sampled_dataset = dataset
        logger.info(f"Analyzing full dataset ({total_rows} samples).")

    logger.info(f"Checking for documents with > {threshold} tokens...")

    def calc_length(batch):
        # Tokenize without truncation/padding to get true length
        encodings = tokenizer(
            batch["text"],
            truncation=False,
            padding=False,
            return_attention_mask=False,
        )
        return {"token_count": [len(ids) for ids in encodings["input_ids"]]}

    dataset_with_lengths = sampled_dataset.map(
        calc_length,
        batched=True,
        num_proc=16,
        desc="Counting tokens",
        remove_columns=["text"],
    )

    lengths = dataset_with_lengths["token_count"]
    total_docs = len(lengths)
    total_tokens = sum(lengths)
    long_docs_count = sum(1 for l in lengths if l > threshold)
    percentage = (long_docs_count / total_docs) * 100 if total_docs > 0 else 0

    logger.info(f"Token distribution analysis (Threshold: {threshold})")
    logger.info(f"Total Documents: {total_docs}")

    logger.info(f"Total Tokens: {total_tokens:_}")
    logger.info(f"Tokens (Billions): {total_tokens / 1_000_000_000:.4f} B")

    logger.info(f"Documents > {threshold} tokens: {long_docs_count}")
    logger.info(f"Percentage: {percentage:.4f}%")
    if total_docs > 0:
        logger.info(f"Max Length: {max(lengths)}")
        logger.info(f"Avg Length: {sum(lengths)/total_docs:.2f}")

    return long_docs_count


def tokenize_dataset_with_padding(
    data_folder, tokenizer, vocabulary_size, context_size, input_dataset
):
    tokenizer.model_max_length = context_size

    logger.info(f"The tokenizer will keep only: {context_size} tokens")

    total_size = len(input_dataset)
    # subset_size = int(total_size * 0.05)

    # logger.info(f"Downsampling: Using 5% of dataset ({subset_size} out of {total_size} samples)")

    # input_dataset = input_dataset.select(range(subset_size))

    def group_texts(examples):
        tokenized_inputs = tokenizer(
            examples["text"],
            max_length=context_size,
            truncation=True,
            padding=False,
            return_special_tokens_mask=True,
        )
        return tokenized_inputs

    # preprocess dataset
    logger.info("Tokenizing Dataset")
    tokenized_datasets = input_dataset.map(
        group_texts,
        batched=True,
        batch_size=50_000,
        remove_columns=["text"],
        num_proc=16,
    )

    logger.info("Finished tokenizing dataset")

    tokenized_datasets_name = os.path.join(
        data_folder,
        f"padded-tokenized-for-training/custom/vocab_size:{vocabulary_size:_}/context_size:{context_size}",
    )
    tokenized_datasets.save_to_disk(tokenized_datasets_name)

    return tokenized_datasets


def create_and_train_tokenizer(
    vocabulary_size, context_size, input_dataset, train_tokenizer=False
):
    tokenizer_path = f"tokenizers/custom/{vocabulary_size:_}"

    if (not train_tokenizer) and os.path.isdir(tokenizer_path):
        logger.info(
            f"Tokenizer already found at {tokenizer_path}. Loading from disk."
        )
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    else:
        logger.info(f"Starting new tokenizer training.")
        tokenizer_handler = TokenizerHandler(
            vocabulary_size, context_size, input_dataset
        )

        logger.info("Training tokenizer with splitted dataset")
        tokenizer_handler.train_tokenizer()
        tokenizer_handler.update_to_fast_tokenizer()

        logger.info("Saving trained tokenier")
        tokenizer = tokenizer_handler.save_tokenizer()

    # analyze_token_distribution(input_dataset["train"], tokenizer, threshold=512)

    return tokenizer
