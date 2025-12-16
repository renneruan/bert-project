import math
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from multiprocessing import cpu_count
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from scipy.stats import pearsonr

# Hugging Face imports
import evaluate
from datasets import load_from_disk, load_dataset
from transformers import (
    AutoTokenizer,
    ModernBertForMaskedLM,
    ModernBertForTokenClassification,
    ModernBertForSequenceClassification,
    DataCollatorForLanguageModeling,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    pipeline,
    get_wsd_schedule,
)
from torch.optim import AdamW


# Configuration & Environment Setup
class Config:
    # Paths
    WORK_DIR = os.getenv("WORK", ".")  # Defaults to current dir if not set
    DATA_FOLDER = os.path.join(WORK_DIR, "data")
    CACHED_DATA_FOLDER = os.path.join(WORK_DIR, "cached_data")

    # Model & Tokenizer Paths
    TOKENIZER_PATH = "/work1/lgarcia/renneruan/tokenizers/custom/32_768"
    # Base checkpoint for fine-tuning
    BASE_MODEL_PATH = "/work1/lgarcia/renneruan/training_test/Modern/classiccc-1024-unigram-32768-900ksteps/checkpoint-900000"

    # Dataset Paths

    LENER_PATH = os.path.join(WORK_DIR, "data", "lener_br_local")
    ASSIN2_PATH = "nilc-nlp/assin2"

    # Hyperparameters
    CONTEXT_SIZE = 1024  # Reduced from 1024 for downstream tasks usually, or keep 1024 if needed
    VOCAB_SIZE = 32_768
    SEED = 42

    TOKENIZED_DS_PATH = os.path.join(
        DATA_FOLDER,
        f"tokenized-for-training/custom/vocab_size:{VOCAB_SIZE:_}/context_size:{CONTEXT_SIZE}",
    )


def setup_environment():
    """Sets up OS environment variables."""
    os.environ["HF_HOME"] = Config.CACHED_DATA_FOLDER
    os.environ["TRITON_HIP_LLD_PATH"] = "/opt/rocm-6.4.1/lib/llvm/bin/ld.lld"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # Disable tokenizers parallelism to avoid deadlocks in dataloaders
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    print(f"Working Directory: {Config.WORK_DIR}")


class Metrics:
    seqeval = evaluate.load("seqeval")

    @staticmethod
    def compute_ner(p, label_list):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = Metrics.seqeval.compute(
            predictions=true_predictions, references=true_labels
        )

        # Flatten metrics
        metrics = {
            "overall_precision": results["overall_precision"],
            "overall_recall": results["overall_recall"],
            "overall_f1": results["overall_f1"],
            "overall_accuracy": results["overall_accuracy"],
        }
        return metrics

    @staticmethod
    def compute_rte(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=1)
        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average="macro")
        return {"accuracy": acc, "macro_f1": f1}

    @staticmethod
    def compute_sts(eval_pred):
        predictions, labels = eval_pred
        predictions = predictions.squeeze()
        labels = labels.squeeze()
        mse = mean_squared_error(labels, predictions)
        pearson_corr, _ = pearsonr(labels, predictions)
        return {"mse": mse, "pearson": pearson_corr}


# MLM Testing
def run_mlm_test(tokenizer):
    model = ModernBertForMaskedLM.from_pretrained(Config.BASE_MODEL_PATH)
    model.to("cuda")

    # Quantitative Evaluation
    print("\nQuantitative Evaluation (Test Set)")

    if os.path.exists(Config.TOKENIZED_DS_PATH):
        print(f"Loading dataset from: {Config.TOKENIZED_DS_PATH}")
        try:
            tokenized_datasets = load_from_disk(Config.TOKENIZED_DS_PATH)
            evaluation_dataset = tokenized_datasets["test"]

            # Using your specific probability of 0.3
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer, mlm=True, mlm_probability=0.3
            )

            eval_args = TrainingArguments(
                output_dir="evaluating/mlm_test",
                do_train=False,
                per_device_eval_batch_size=32,
                logging_dir="evaluating/evaluation-logs",
                report_to=["tensorboard"],
                fp16=True,
            )

            evaluator = Trainer(
                model=model,
                args=eval_args,
                eval_dataset=evaluation_dataset,
                data_collator=data_collator,
            )

            print(f"Evaluating on {len(evaluation_dataset)} samples...")
            eval_results = evaluator.evaluate()

            print("Evaluation Results:", eval_results)

            # Calculate Perplexity manually for clarity
            if "eval_loss" in eval_results:
                perplexity = math.exp(eval_results["eval_loss"])
                print(f"Perplexity: {perplexity:.4f}")

        except Exception as e:
            print(f"Failed to load or evaluate dataset: {e}")
    else:
        print(
            f"Warning: Tokenized dataset not found at {Config.TOKENIZED_DS_PATH}. Skipping quantitative eval."
        )

    # Qualitative Prediction (Fill-Mask)
    print("\n>>> Phase 2: Qualitative Prediction (Fill-Mask)")
    fill_mask = pipeline(
        "fill-mask", model=model, tokenizer=tokenizer, device=0
    )

    test_sentences = [
        "Eu não entendi [MASK], como proceder.",
        "Atiraram o pau no [MASK]",
        "Essa frase tem [MASK] tokens, portanto ele vai gerar dois [MASK]",
    ]

    for sent in test_sentences:
        print(f"\nInput: {sent}")
        res = fill_mask(sent)

        # Handle cases where multiple masks create list of lists
        if isinstance(res, list) and len(res) > 0 and isinstance(res[0], list):
            for i, r in enumerate(res):
                top_token = r[0]["token_str"]
                print(f"  Mask {i+1}: {top_token} (conf: {r[0]['score']:.2f})")
        elif isinstance(res, list):
            top_token = res[0]["token_str"]
            print(f"  Prediction: {top_token} (conf: {res[0]['score']:.2f})")


# NER
def run_ner(tokenizer):
    print("\n--- Running NER Fine-tuning ---")
    dataset = load_from_disk(Config.LENER_PATH)
    label_list = dataset["train"].features["ner_tags"].feature.names

    # Mappings
    id2label = {i: label for i, label in enumerate(label_list)}
    label2id = {label: i for i, label in enumerate(label_list)}

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"],
            truncation=True,
            padding="max_length",
            max_length=Config.CONTEXT_SIZE,
            is_split_into_words=True,
        )
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    encoded_dataset = dataset.map(
        tokenize_and_align_labels, batched=True, num_proc=cpu_count()
    )
    encoded_dataset = encoded_dataset.remove_columns(
        ["id", "tokens", "ner_tags"]
    )

    model = ModernBertForTokenClassification.from_pretrained(
        Config.BASE_MODEL_PATH,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    args = TrainingArguments(
        output_dir=f"trained/NER/modern-bert-ner",
        max_steps=3000,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=32,
        save_strategy="steps",
        save_steps=100,
        eval_strategy="steps",
        eval_steps=100,
        logging_strategy="steps",
        logging_steps=100,
        learning_rate=2e-5,
        weight_decay=1e-4,
        load_best_model_at_end=True,
        metric_for_best_model="overall_f1",
        greater_is_better=True,
        report_to=["tensorboard"],
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    lr_scheduler = get_wsd_schedule(
        optimizer=optimizer,
        num_warmup_steps=int(args.max_steps * 0.1),
        num_stable_steps=int(args.max_steps * 0.3),
        num_decay_steps=int(args.max_steps * 0.6),
        min_lr_ratio=0,
        num_cycles=0.5,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
        data_collator=DataCollatorForTokenClassification(
            tokenizer=tokenizer, padding=True
        ),
        compute_metrics=lambda p: Metrics.compute_ner(p, label_list),
        optimizers=(optimizer, lr_scheduler),
    )

    trainer.train()
    print("Evaluating on Test Set...")
    print(trainer.evaluate(encoded_dataset["test"]))


# RTE
def run_rte(tokenizer):
    print("\n--- Running RTE Fine-tuning ---")
    dataset = load_dataset(Config.ASSIN2_PATH)

    def preprocess(example):
        return tokenizer(
            example["premise"],
            example["hypothesis"],
            truncation=True,
            padding="max_length",
            max_length=Config.CONTEXT_SIZE,
        )

    encoded_dataset = dataset.map(preprocess, batched=True)
    encoded_dataset = encoded_dataset.rename_column(
        "entailment_judgment", "labels"
    )
    encoded_dataset = encoded_dataset.remove_columns(
        ["sentence_pair_id", "relatedness_score", "premise", "hypothesis"]
    )
    encoded_dataset.set_format("torch")

    model = ModernBertForSequenceClassification.from_pretrained(
        Config.BASE_MODEL_PATH, num_labels=2
    )

    args = TrainingArguments(
        output_dir=f"trained/RTE/modern-bert-rte",
        max_steps=10000,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        save_strategy="steps",
        save_steps=100,
        eval_strategy="steps",
        eval_steps=100,
        logging_steps=100,
        learning_rate=1e-5,
        weight_decay=1e-4,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        report_to=["tensorboard"],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
        compute_metrics=Metrics.compute_rte,
    )

    trainer.train()
    print("Evaluating on Test Set...")
    print(trainer.evaluate(encoded_dataset["test"]))


# STS
def run_sts(tokenizer):
    print("\n--- Running STS Fine-tuning ---")
    dataset = load_dataset(Config.ASSIN2_PATH)

    def preprocess(example):
        return tokenizer(
            example["premise"],
            example["hypothesis"],
            truncation=True,
            padding="max_length",
            max_length=Config.CONTEXT_SIZE,
        )

    encoded_dataset = dataset.map(preprocess, batched=True)
    encoded_dataset = encoded_dataset.rename_column(
        "relatedness_score", "labels"
    )
    encoded_dataset = encoded_dataset.remove_columns(
        ["sentence_pair_id", "entailment_judgment", "premise", "hypothesis"]
    )
    encoded_dataset.set_format("torch")

    model = ModernBertForSequenceClassification.from_pretrained(
        Config.BASE_MODEL_PATH, num_labels=1
    )

    args = TrainingArguments(
        output_dir=f"trained/STS/modern-bert-sts",
        max_steps=5000,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        save_strategy="steps",
        save_steps=100,
        eval_strategy="steps",
        eval_steps=100,
        logging_steps=100,
        learning_rate=2e-5,
        weight_decay=1e-4,
        load_best_model_at_end=True,
        metric_for_best_model="mse",
        greater_is_better=False,
        report_to=["tensorboard"],
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    lr_scheduler = get_wsd_schedule(
        optimizer=optimizer,
        num_warmup_steps=int(args.max_steps * 0.1),
        num_stable_steps=int(args.max_steps * 0.2),
        num_decay_steps=int(args.max_steps * 0.7),
        min_lr_ratio=0,
        num_cycles=0.5,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
        compute_metrics=Metrics.compute_sts,
        optimizers=(optimizer, lr_scheduler),
    )

    trainer.train()
    print("Evaluating on Test Set...")
    print(trainer.evaluate(encoded_dataset["test"]))


# Main Execution
if __name__ == "__main__":
    setup_environment()

    parser = argparse.ArgumentParser(
        description="ModernBERT Fine-tuning Pipeline"
    )
    parser.add_argument(
        "task", choices=["mlm", "ner", "rte", "sts"], help="Task to execute"
    )
    args = parser.parse_args()

    # Load tokenizer once
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            Config.TOKENIZER_PATH,
            local_files_only=True,
            cache_dir=Config.CACHED_DATA_FOLDER,
        )
    except OSError:
        print(
            f"Error: Tokenizer not found at {Config.TOKENIZER_PATH}. Check path."
        )
        sys.exit(1)

    # Route to task
    if args.task == "mlm":
        run_mlm_test(tokenizer)
    elif args.task == "ner":
        run_ner(tokenizer)
    elif args.task == "rte":
        run_rte(tokenizer)
    elif args.task == "sts":
        run_sts(tokenizer)
