import os
import torch
from datasets import load_from_disk
from transformers import (
    ModernBertConfig,
    ModernBertForMaskedLM,
    DataCollatorForLanguageModeling,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

from accelerate import Accelerator


def run_training():

    accelerator = Accelerator()

    WORK_DIR = os.getenv("WORK")
    DATA_FOLDER = os.path.join(WORK_DIR, "data")
    CACHED_DATA_FOLDER = os.path.join(WORK_DIR, "cached_data")
    os.environ["HF_HOME"] = CACHED_DATA_FOLDER
    os.environ["TRITON_HIP_LLD_PATH"] = "/opt/rocm-6.4.1/lib/llvm/bin/ld.lld"
    os.chdir(WORK_DIR)

    accelerator.print(f"Working directory: {os.getcwd()}")

    vocabulary_size = 32_768
    context_size = 1024
    tokenizer_name = f"tokenizers/custom/{vocabulary_size:_}"
    model_name = f"Modern/classiccc-1024-unigram-32768-900ksteps"

    output_dir = f"training_test/{model_name}"
    os.makedirs(output_dir, exist_ok=True)

    tokenized_datasets_name = os.path.join(
        DATA_FOLDER,
        f"padded-tokenized-for-training/custom/vocab_size:{vocabulary_size:_}/context_size:{context_size}",
    )
    tokenized_datasets = load_from_disk(tokenized_datasets_name)
    training_dataset = tokenized_datasets
    # eval_dataset = tokenized_datasets["test"]

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, local_files_only=True, cache_dir=CACHED_DATA_FOLDER
    )

    config = ModernBertConfig.from_pretrained(
        "answerdotai/ModernBERT-base",
        reference_compile=False,
        attn_implementation="flash_attention_2",
        vocab_size=vocabulary_size,
        max_position_embeddings=1024,
    )
    config.vocab_size = vocabulary_size
    config.max_position_embeddings = 1024

    config.global_rope_theta = 10000.0

    config.local_attention = 128
    config.pad_token_id = 0
    config.bos_token_id = 2
    config.cls_token_id = 2
    config.eos_token_id = 3
    config.sep_token_id = 3

    model = ModernBertForMaskedLM(config=config)
    # NOTE: We do NOT call model.to("cuda") or model.half().
    # The Trainer, powered by Accelerate, will handle device placement and mixed precision.

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.3
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=False,
        max_steps=900_000,
        per_device_train_batch_size=256,
        gradient_accumulation_steps=1,
        dataloader_num_workers=128,
        logging_strategy="steps",
        logging_first_step=True,
        logging_steps=1_000,
        save_strategy="steps",
        save_steps=1_000,
        save_total_limit=5,
        fp16=True,
        report_to="tensorboard",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=training_dataset,
        data_collator=data_collator,
    )

    accelerator.print("Starting training on all available GPUs...")
    last_checkpoint = None
    if os.path.isdir(output_dir):
        last_checkpoint = get_last_checkpoint(output_dir)

    if last_checkpoint is not None:
        accelerator.print(f"Resuming from checkpoint: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        accelerator.print("No checkpoint found. Starting from scratch.")
        trainer.train()
    accelerator.print("Training complete!")


if __name__ == "__main__":
    run_training()
