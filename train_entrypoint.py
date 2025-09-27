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
from accelerate import Accelerator

def run_training():

    # Initialize Accelerator. The Trainer will automatically detect and use it.
    accelerator = Accelerator()

    WORK_DIR = os.getenv('WORK')
    DATA_FOLDER = os.path.join(WORK_DIR, "data")
    CACHED_DATA_FOLDER = os.path.join(WORK_DIR, "cached_data")
    os.environ['HF_HOME'] = CACHED_DATA_FOLDER
    os.environ['TRITON_HIP_LLD_PATH'] = '/opt/rocm-6.4.1/lib/llvm/bin/ld.lld'
    os.chdir(WORK_DIR)
    
    accelerator.print(f"Working directory: {os.getcwd()}")

    vocabulary_size = 32_768
    context_size = 512
    tokenizer_name = f"tokenizers/custom/{vocabulary_size:_}"
    model_name = f"Modern/{4.6}"

    tokenized_datasets_name = os.path.join(DATA_FOLDER, f"tokenized-for-training/custom/vocab_size:{vocabulary_size:_}/context_size:{context_size}")
    tokenized_datasets = load_from_disk(tokenized_datasets_name)
    training_dataset = tokenized_datasets["train"]
    
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        local_files_only=True,
        cache_dir=CACHED_DATA_FOLDER
    )

    config = ModernBertConfig.from_pretrained(
        "answerdotai/ModernBERT-base",
        reference_compile=False,
        attn_implementation="flash_attention_2",
    )
    config.vocab_size = vocabulary_size
    config.max_position_embeddings = 512
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
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.3
    )

    training_args = TrainingArguments(
        output_dir=f'training/{model_name}',
        overwrite_output_dir=True,
        max_steps=500_000,
        per_device_train_batch_size=256,   
        gradient_accumulation_steps=1,
        dataloader_num_workers=64,         
        logging_strategy="steps",
        logging_first_step=True,
        logging_steps=1_000,
        save_strategy="steps",
        save_steps=1_000,
        save_total_limit=5,
        fp16=True,             
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=training_dataset,
        data_collator=data_collator,
    )

    accelerator.print("Starting training on all available GPUs...")
    trainer.train()
    accelerator.print("Training complete!")
    
if __name__ == "__main__":
    run_training()