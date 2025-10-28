import os
from datasets import load_from_disk
from transformers import AutoTokenizer, ModernBertConfig, ModernBertForMaskedLM
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
import torch

vocabulary_size = 32_768
context_size = 512
os.environ["HF_HOME"] = "cache/"
default_cache_dir = "cache/"
model_name = f"Modern/{5.0}"

def load_dataset():
    tokenized_datasets_name = f"dataset/tokenized-for-training/custom/vocab_size:{vocabulary_size:_}/context_size:{context_size}"
    tokenized_datasets = load_from_disk(tokenized_datasets_name)
    training_dataset = tokenized_datasets["train"]
    return training_dataset

def load_tokenizer():
    tokenizer_name = f"tokenizers/custom/{vocabulary_size:_}"
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, 
        local_files_only=True, 
        cache_dir = default_cache_dir
    )
    return tokenizer

def load_collator(tokenizer):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm = True,
        mlm_probability=0.3
    )
    return data_collator

def load_model():
    model = ModernBertForMaskedLM.from_pretrained(f"models/{model_name}", local_files_only=True)
    model.to("cuda")
    return model

def load_trainer(model, data_collator, training_dataset):
    total_steps = 500_000

    training_args = TrainingArguments(
        output_dir=f'training/{model_name}',
        overwrite_output_dir=True,
        
        # num_train_epochs=1,                     # number of training epochs
        max_steps=total_steps,
        # max_steps=100,

        gradient_accumulation_steps = 1,
        # eval_accumulation_steps = 1,

        per_device_train_batch_size=32,          # batch size for training
        # per_device_eval_batch_size=32,           # batch size for evaluation

        logging_strategy="steps",
        logging_first_step=True, # output the initial loss
        logging_steps=1_000,
        logging_dir=f"training-logs/{model_name}",
        report_to=["tensorboard"],

        save_strategy="steps",
        save_steps=1_000,                      # Save checkpoints every 100 steps
        save_total_limit=5,                  # Limit the total number of saved checkpoints

        fp16=True,                            # Enable mixed precision for faster training

        # learning_rate=8e-4,
        # weight_decay=1e-2,
        # adam_beta1=0.9,
        # adam_beta2=0.999,
        # adam_epsilon=1e-06,
        # lr_scheduler_type=
    )

    trainer = Trainer(
        model=model,                        # Model to train
        args=training_args,                 # Training arguments
        train_dataset=training_dataset,     # Training dataset
        data_collator=data_collator,
    )

    return trainer

# main
dataset = load_dataset()
tokenizer = load_tokenizer()
collator = load_collator(tokenizer)
model = load_model()
trainer = load_trainer(model, collator, dataset)

torch.cuda.empty_cache()  # Clear cached memory

trainer.train(resume_from_checkpoint=True)