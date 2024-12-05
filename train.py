import os
import torch
from torch.nn.utils import clip_grad_norm_
from transformers import BartForConditionalGeneration, BartTokenizer
from torch.utils.data import DataLoader, TensorDataset
import itertools
from typing import List, Tuple, Dict
import random
from tqdm import tqdm
from auxiliary import ARCData

BATCH_SIZE = 16
EPOCHS = 10
SAVE_PATH = "saved_model/checkpoint.pth"

model_name = "facebook/bart-base"

tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_train(dataset, tokenizer, max_length=1024, max_permutations=4):
    """
    Prepare batches keeping task relationships intact while permuting examples.

    Args:
        dataset: List of (task_ex, task_ts) pairs where each task contains multiple input-output pairs
        tokenizer: Tokenizer for encoding strings
        max_length: Maximum sequence length
        max_permutations: Maximum number of permutations to generate per task
    """

    def matrix_to_string(matrix: List[List[int]]) -> str:
        """Convert a matrix to its string representation"""
        rows = [f"[{' '.join(str(x) for x in row)}]" for row in matrix]
        return f"[{' '.join(rows)}]"

    data = []

    for task_ex, task_ts in tqdm(dataset):
        all_ex = list(task_ex)
        all_ex.extend(list(task_ts))

        for target_pair in all_ex:
            target_input = target_pair["input"]
            target_output = target_pair["output"]

            target_input_str = matrix_to_string(target_input)
            target_output_str = matrix_to_string(target_output)

            other_examples = []
            for other_pair in all_ex:
                if other_pair != target_pair:
                    input_str = matrix_to_string(other_pair["input"])
                    output_str = matrix_to_string(other_pair["output"])
                    other_examples.append(f"{input_str} [IO] {output_str}")

            max_examples = 0
            for num_examples in range(1, len(other_examples) + 1):
                test_str = f"[CLS] {target_input_str} [IO] [SEP] [CLS] {' [EX] '.join(other_examples[:num_examples])} [SEP]"
                if (
                    len(tokenizer.encode(test_str, add_special_tokens=False))
                    >= max_length
                ):
                    max_examples = num_examples - 1
                    break
                max_examples = num_examples

            if max_examples < 1:
                continue

            # Trim examples to max that fits
            other_examples = other_examples[:max_examples]

            # Generate permutations of example pairs
            all_perms = list(itertools.permutations(other_examples))

            # If there are too many permutations, randomly sample max_permutations
            if len(all_perms) > max_permutations:
                all_perms = random.sample(all_perms, max_permutations)

            # Create input-output pairs for each permutation
            for perm in all_perms:
                example_str = " [EX] ".join(perm)

                # Create input_ids
                input_combined_str = (
                    f"[CLS] {target_input_str} [IO] [SEP] [CLS] {example_str} [SEP]"
                )
                input_ids = tokenizer.encode(
                    input_combined_str,
                    add_special_tokens=False,
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                )
                input_ids = torch.tensor([input_ids])

                # Create output_ids (using target output from same pair)
                output_combined_str = f"[CLS] {target_output_str} [SEP]"
                output_ids = tokenizer.encode(
                    output_combined_str,
                    add_special_tokens=False,
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                )
                output_ids = torch.tensor([output_ids])

                data.append((input_ids, output_ids))

    return data


def prepare_eval(dataset, tokenizer, max_length=1024):
    """
    Prepare batches keeping task relationships intact while permuting examples.

    Args:
        dataset: List of (task_ex, task_ts) pairs where each task contains multiple input-output pairs
        tokenizer: Tokenizer for encoding strings
        max_length: Maximum sequence length

    """

    def matrix_to_string(matrix: List[List[int]]) -> str:
        """Convert a matrix to its string representation"""
        rows = [f"[{' '.join(str(x) for x in row)}]" for row in matrix]
        return f"[{' '.join(rows)}]"

    data = []

    for task_ex, task_ts in tqdm(dataset):
        all_ex = list(task_ex)
        all_ts = list(task_ts)

        for target_pair in all_ts:
            target_input = target_pair["input"]
            target_output = target_pair["output"]

            target_input_str = matrix_to_string(target_input)
            target_output_str = matrix_to_string(target_output)

            other_examples = []
            for other_pair in all_ex:
                input_str = matrix_to_string(other_pair["input"])
                output_str = matrix_to_string(other_pair["output"])
                other_examples.append(f"{input_str} [IO] {output_str}")

            max_examples = 0
            for num_examples in range(1, len(other_examples) + 1):
                test_str = f"[CLS] {target_input_str} [IO] [SEP] [CLS] {' [EX] '.join(other_examples[:num_examples])} [SEP]"
                if (
                    len(tokenizer.encode(test_str, add_special_tokens=False))
                    >= max_length
                ):
                    max_examples = num_examples - 1
                    break
                max_examples = num_examples

            if max_examples < 1:
                continue

            example_str = " [EX] ".join(other_examples)

            # Create input_ids
            input_combined_str = (
                f"[CLS] {target_input_str} [IO] [SEP] [CLS] {example_str} [SEP]"
            )
            input_ids = tokenizer.encode(
                input_combined_str,
                add_special_tokens=False,
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )
            input_ids = torch.tensor([input_ids])

            # Create output_ids (using target output from same pair)
            output_combined_str = f"[CLS] {target_output_str} [SEP]"
            output_ids = tokenizer.encode(
                output_combined_str,
                add_special_tokens=False,
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )
            output_ids = torch.tensor([output_ids])

            data.append((input_ids, output_ids))

    return data


def create_train_dataloader(
    dataset,
    tokenizer,
    batch_size=BATCH_SIZE,
    max_length=1024,
    max_permutations=4,
    shuffle=True,
):
    """
    Create a DataLoader with permuted batches.
    """

    data = prepare_train(
        dataset, tokenizer, max_length=max_length, max_permutations=max_permutations
    )

    # Combine all batches into tensors
    all_input_ids = torch.cat([b[0] for b in data], dim=0)
    all_output_ids = torch.cat([b[1] for b in data], dim=0)

    # Create dataset and dataloader
    tensor_dataset = TensorDataset(all_input_ids, all_output_ids)
    dataloader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader


def create_eval_dataloader(dataset, tokenizer, batch_size=BATCH_SIZE, max_length=1024):
    """
    Create a DataLoader with permuted batches.
    """

    data = prepare_eval(dataset, tokenizer, max_length=max_length)

    # Combine all batches into tensors
    all_input_ids = torch.cat([b[0] for b in data], dim=0)
    all_output_ids = torch.cat([b[1] for b in data], dim=0)

    # Create dataset and dataloader
    tensor_dataset = TensorDataset(all_input_ids, all_output_ids)
    dataloader = DataLoader(tensor_dataset, batch_size=batch_size)

    return dataloader


def train_model(
    model,
    train_loader,
    eval_loader,
    optimizer,
    scheduler,
    device,
    num_epochs=3,
    max_grad_norm=1.0,
    save_path=SAVE_PATH,
):
    model.to(device)
    try:
        for epoch in range(num_epochs):
            # Training Phase
            model.train()
            total_train_loss = 0.0
            total_train_correct = 0
            total_train_tokens = 0

            for batch in tqdm(
                train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training"
            ):
                optimizer.zero_grad()

                input_ids = batch[0].to(device)
                labels = batch[1].to(device)

                # Forward pass
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                logits = outputs.logits

                # Compute token-wise accuracy
                predictions = torch.argmax(logits, dim=-1)
                mask = labels != 1
                correct = (predictions == labels) & mask
                total_train_correct += correct.sum().item()
                total_train_tokens += mask.sum().item()

                # Backward pass and optimization
                loss.backward()
                clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            train_accuracy = total_train_correct / total_train_tokens
            print(
                f"Epoch {epoch + 1}/{num_epochs} - Average Training Loss: {avg_train_loss:.4f} - "
                f"Training Accuracy: {train_accuracy:.4f}"
            )
            torch.cuda.empty_cache()

            # Evaluation Phase
            model.eval()
            total_eval_loss = 0.0
            total_eval_correct = 0
            total_eval_tokens = 0

            with torch.no_grad():
                for batch in tqdm(
                    eval_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Evaluating"
                ):
                    input_ids = batch[0].to(device)
                    labels = batch[1].to(device)

                    # Forward pass
                    outputs = model(input_ids=input_ids, labels=labels)
                    loss = outputs.loss
                    logits = outputs.logits

                    # Compute token-wise accuracy
                    predictions = torch.argmax(logits, dim=-1)
                    mask = labels != 1
                    correct = (predictions == labels) & mask
                    total_eval_correct += correct.sum().item()
                    total_eval_tokens += mask.sum().item()

                    total_eval_loss += loss.item()

            avg_eval_loss = total_eval_loss / len(eval_loader)
            eval_accuracy = total_eval_correct / total_eval_tokens
            print(
                f"Epoch {epoch + 1}/{num_epochs} - Average Evaluation Loss: {avg_eval_loss:.4f} - "
                f"Evaluation Accuracy: {eval_accuracy:.4f}"
            )

            # Adjust learning rate
            scheduler.step()
            torch.cuda.empty_cache()

        print("Training complete!")

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving model...")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}.")


train_set = ARCData("train")
eval_set = ARCData("eval")

train_loader = create_train_dataloader(train_set, tokenizer)
eval_loader = create_eval_dataloader(eval_set, tokenizer)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

# Run training loop
train_model(
    model,
    train_loader,
    eval_loader,
    optimizer,
    scheduler,
    device,
    num_epochs=EPOCHS,
    max_grad_norm=1.0,
)
