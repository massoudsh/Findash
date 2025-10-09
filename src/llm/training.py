import gc
import os
import sys
import threading
import time

import psutil
import torch
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup, set_seed

from peft import get_peft_model

from . import config
from .utils import b2mb

# This context manager is used to track the peak memory usage of the process
class TorchTracemalloc:
    def __enter__(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()  # reset the peak gauge to zero
        self.begin = torch.cuda.memory_allocated()
        self.process = psutil.Process()

        self.cpu_begin = self.cpu_mem_used()
        self.peak_monitoring = True
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start()
        return self

    def cpu_mem_used(self):
        """get resident set size memory for the current process"""
        return self.process.memory_info().rss

    def peak_monitor_func(self):
        self.cpu_peak = -1

        while True:
            self.cpu_peak = max(self.cpu_mem_used(), self.cpu_peak)
            if not self.peak_monitoring:
                break

    def __exit__(self, *exc):
        self.peak_monitoring = False

        gc.collect()
        torch.cuda.empty_cache()
        self.end = torch.cuda.memory_allocated()
        self.peak = torch.cuda.max_memory_allocated()
        self.used = b2mb(self.end - self.begin)
        self.peaked = b2mb(self.peak - self.begin)

        self.cpu_end = self.cpu_mem_used()
        self.cpu_used = b2mb(self.cpu_end - self.cpu_begin)
        self.cpu_peaked = b2mb(self.cpu_peak - self.cpu_begin)
        print(f"CPU Mem aktuell: {self.cpu_used} MB")
        print(f"CPU Mem spitze: {self.cpu_peaked} MB")
        print(f"GPU Mem aktuell: {self.used} MB")
        print(f"GPU Mem spitze: {self.peaked} MB")


def levenshtein_distance(str1, str2):
    # TC: O(N^2)
    # SC: O(N)
    if str1 == str2:
        return 0
    num_rows = len(str1) + 1
    num_cols = len(str2) + 1
    dp_matrix = list(range(num_cols))
    for i in range(1, num_rows):
        prev = dp_matrix[0]
        dp_matrix[0] = i
        for j in range(1, num_cols):
            temp = dp_matrix[j]
            if str1[i - 1] == str2[j - 1]:
                dp_matrix[j] = prev
            else:
                dp_matrix[j] = min(prev, dp_matrix[j], dp_matrix[j - 1]) + 1
            prev = temp
    return dp_matrix[num_cols - 1]


def get_closest_label(eval_pred, classes):
    min_id = sys.maxsize
    min_edit_distance = sys.maxsize
    for i, class_label in enumerate(classes):
        edit_distance = levenshtein_distance(eval_pred.strip(), class_label)
        if edit_distance < min_edit_distance:
            min_id = i
            min_edit_distance = edit_distance
    return classes[min_id]


def run_training(do_test: bool = False, output_dir: str = "./peft-output"):
    accelerator = Accelerator()
    set_seed(config.SEED)

    # Load dataset
    dataset = load_dataset(config.RAFT_DATASET_SUBSET, config.DATASET_NAME)
    classes = [k.replace("_", " ") for k in dataset["train"].features["Label"].names]
    dataset = dataset.map(
        lambda x: {"text_label": [classes[label] for label in x["Label"]]},
        batched=True,
        num_proc=1,
    )

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME_OR_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.MODEL_NAME_OR_PATH)
    model = get_peft_model(model, config.PEFT_CONFIG)
    model.print_trainable_parameters()

    def preprocess_function(examples):
        inputs = examples[config.TEXT_COLUMN]
        targets = examples[config.LABEL_COLUMN]
        model_inputs = tokenizer(inputs, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
        labels = tokenizer(targets, max_length=3, padding="max_length", truncation=True, return_tensors="pt")
        labels = labels["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels
        return model_inputs

    with accelerator.main_process_first():
        processed_datasets = dataset.map(
            preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=dataset["train"].column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["test"]

    def collate_fn(examples):
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=config.BATCH_SIZE, pin_memory=True
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=collate_fn, batch_size=config.BATCH_SIZE, pin_memory=True)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * config.NUM_EPOCHS),
    )

    # Prepare for distributed training
    model, train_dataloader, eval_dataloader, optimizer, lr_scheduler = accelerator.prepare(
        model, train_dataloader, eval_dataloader, optimizer, lr_scheduler
    )

    # Training loop
    for epoch in range(config.NUM_EPOCHS):
        with TorchTracemalloc() as tracemalloc:
            model.train()
            total_loss = 0
            for step, batch in enumerate(tqdm(train_dataloader)):
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

        # Evaluation loop
        model.eval()
        eval_loss = 0
        eval_preds = []
        for step, batch in enumerate(tqdm(eval_dataloader)):
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            eval_loss += loss.detach().float()
            
            # Generate predictions
            pred_ids = accelerator.unwrap_model(model).generate(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], max_new_tokens=10
            )
            eval_preds.extend(tokenizer.batch_decode(pred_ids, skip_special_tokens=True))

        eval_epoch_loss = eval_loss / len(eval_dataloader)
        eval_ppl = torch.exp(eval_epoch_loss)
        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")
        
        # Compute accuracy
        correct = 0
        total = 0
        for pred, true in zip(eval_preds, dataset["test"][config.LABEL_COLUMN]):
            if pred.strip() == true.strip():
                correct += 1
            total += 1
        accuracy = correct / total * 100
        print(f"{accuracy=}")

        # For the last epoch, compute the levenshtein distance based accuracy
        if epoch == config.NUM_EPOCHS - 1:
            correct = 0
            total = 0
            for pred, true in zip(eval_preds, dataset["test"][config.LABEL_COLUMN]):
                pred_label = get_closest_label(pred, classes)
                if pred_label == true.strip():
                    correct += 1
                total += 1
            accuracy = correct / total * 100
            print(f"Levenshtein Accuracy: {accuracy=}")

    # Save the model
    if output_dir:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(output_dir)

    # Optionally run test set evaluation
    if do_test:
        model.eval()
        test_preds = []
        test_dataloader = DataLoader(processed_datasets["test"], collate_fn=collate_fn, batch_size=config.BATCH_SIZE)
        test_dataloader = accelerator.prepare(test_dataloader)
        for _, batch in enumerate(tqdm(test_dataloader)):
            with torch.no_grad():
                pred_ids = accelerator.unwrap_model(model).generate(
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], max_new_tokens=10
                )
                test_preds.extend(tokenizer.batch_decode(pred_ids, skip_special_tokens=True))
        
        # Save test predictions
        if output_dir:
            output_test_preds_file = os.path.join(output_dir, "test_preds.txt")
            with open(output_test_preds_file, "w") as writer:
                writer.write("\n".join(test_preds))

    print("Training finished successfully.") 