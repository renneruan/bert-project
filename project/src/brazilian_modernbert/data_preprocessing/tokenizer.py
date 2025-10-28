import os
import logging
from datasets import load_from_disk

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
from tokenizers.decoders import Metaspace
from tokenizers.models import Unigram
from tokenizers.trainers import UnigramTrainer

from tqdm import tqdm

from multiprocessing import cpu_count

logger = logging.getLogger(__name__)

CUSTOM_SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


class Tokenizer:
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
        custom_tokenizer.decoder = Metaspace()

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
    def normalize_and_pre_tokenize(text):
        normalized = tokenizer.backend_tokenizer.normalizer.normalize_str(text)
        processed = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(
            normalized
        )
        return processed

    def count_tokens(batch):
        original_tokens = 0
        generated_tokens = 0

        for doc in batch["text"]:

            original_tokens += len(normalize_and_pre_tokenize(doc))
            generated_tokens += len(tokenizer.encode(doc))

        # Add the token counts as a new column to the batch
        return {"generated": [generated_tokens], "original": [original_tokens]}

    evaluate_fertility = dataset["train"].map(
        count_tokens,
        batched=True,
        remove_columns=["text"],
        num_proc=cpu_count(),
    )

    logger.info("Tokenizer Fertility: %s", evaluate_fertility)
    return evaluate_fertility


def tokenize_dataset(
    data_folder, tokenizer, vocabulary_size, context_size, input_dataset
):
    tokenizer.model_max_length = context_size

    logger.info(f"The tokenizer will keep only: {context_size} tokens")

    def group_texts(examples):
        tokenized_inputs = tokenizer(
            examples["text"],
            max_length=context_size,
            truncation=True,
            padding="max_length",
            return_special_tokens_mask=True,
        )
        return tokenized_inputs

    # preprocess dataset
    logger.info("Tokenizing Dataset")
    tokenized_datasets = input_dataset.map(
        group_texts,
        batched=True,
        remove_columns=["text"],
        num_proc=cpu_count(),
    )

    logger.info("Finished tokenizing dataset")

    tokenized_datasets_name = os.path.join(
        data_folder,
        f"tokenized-for-training/custom/vocab_size:{vocabulary_size:_}/context_size:{context_size}",
    )
    tokenized_datasets.save_to_disk(tokenized_datasets_name)

    return tokenized_datasets


def create_and_train_tokenizer(vocabulary_size, context_size, input_dataset):
    logger.info("Creating and configuring tokenizer")
    tokenizer_handler = Tokenizer(vocabulary_size, context_size, input_dataset)

    logger.info("Training tokenizer with splitted dataset")
    tokenizer_handler.train_tokenizer()
    tokenizer_handler.update_to_fast_tokenizer()

    logger.info("Saving trained tokenier")
    tokenizer = tokenizer_handler.save_tokenizer()

    return tokenizer
