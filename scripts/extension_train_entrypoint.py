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
from torch.optim import AdamW
from transformers import get_wsd_schedule
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
    context_size = 8192

    source_checkpoint = os.path.join(
        WORK_DIR,
        "training_test/Modern/classiccc-1024-unigram-32768-900ksteps/checkpoint-900000",
    )
    tokenizer_name = f"tokenizers/custom/{vocabulary_size:_}"
    model_name = f"Modern/classiccc-8192-unigram-32768-150ksteps"

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

    config = ModernBertConfig.from_pretrained(source_checkpoint)

    config.vocab_size = vocabulary_size
    config.max_position_embeddings = context_size

    config.global_rope_theta = 160000.0

    config.attn_implementation = "flash_attention_2"

    accelerator.print(
        f"Loading model from {source_checkpoint} with context size {context_size}..."
    )

    config.local_attention = 128
    config.pad_token_id = 0
    config.bos_token_id = 2
    config.cls_token_id = 2
    config.eos_token_id = 3
    config.sep_token_id = 3

    model = ModernBertForMaskedLM.from_pretrained(
        source_checkpoint, config=config, ignore_mismatched_sizes=True
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.3
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=False,
        max_steps=150_000,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        dataloader_num_workers=16,
        logging_strategy="steps",
        logging_first_step=True,
        logging_steps=100,
        save_strategy="steps",
        save_steps=5_000,
        save_total_limit=5,
        fp16=True,
        report_to="tensorboard",
        gradient_checkpointing=True,
        seed=42,
        data_seed=42,
        learning_rate=5e-5,
        weight_decay=0.01,
    )

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=training_args.learning_rate
    )

    total_steps = training_args.max_steps

    lr_scheduler = get_wsd_schedule(
        optimizer=optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_stable_steps=int(total_steps * 0.5),
        num_decay_steps=int(total_steps * 0.4),
        min_lr_ratio=0.0,
        num_cycles=0.5,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=training_dataset,
        data_collator=data_collator,
        optimizers=(optimizer, lr_scheduler),
    )

    accelerator.print("Starting training on all available GPUs...")

    last_checkpoint = get_last_checkpoint(output_dir)
    if os.path.isdir(output_dir):
        last_checkpoint = get_last_checkpoint(output_dir)

    if last_checkpoint is not None:
        accelerator.print(f"Resuming 8k training from: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        accelerator.print(
            "Starting 8k training (initialized from 1024 checkpoint)..."
        )
        trainer.train()

    accelerator.print("Training complete!")


if __name__ == "__main__":
    run_training()
