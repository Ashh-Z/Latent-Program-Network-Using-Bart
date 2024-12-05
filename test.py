import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Tuple, Dict
from tqdm import tqdm
from auxiliary import ARCData

BATCH_SIZE = 16

model_name = "facebook/bart-base"
SAVE_PATH = "saved_model/checkpoint.pth"

tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)
model.load_state_dict(torch.load(SAVE_PATH, weights_only=True))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def infer_model(
    model,
    eval_loader,
    device,
):
    """
    Evaluate the model on a dataset using token-wise accuracy and count perfect predictions.
    """
    # Evaluation Phase
    model.to(device)
    model.eval()
    total_eval_loss = 0.0
    total_eval_correct = 0
    total_eval_tokens = 0
    perfect_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(eval_loader):
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

            # Compute perfect predictions
            for pred, label in zip(predictions, labels):
                pred_tokens = pred[mask[0]].tolist()
                label_tokens = label[mask[0]].tolist()
                if pred_tokens == label_tokens:
                    perfect_predictions += 1
            total_samples += input_ids.size(0)

            total_eval_loss += loss.item()
    eval_accuracy = total_eval_correct / total_eval_tokens

    print(f"Evaluation Accuracy: {eval_accuracy:.4f}")
    print(f"Perfect Predictions: {perfect_predictions}")
    torch.cuda.empty_cache()


eval_set = ARCData("eval")
eval_loader = create_eval_dataloader(eval_set, tokenizer)

# Run training loop
infer_model(
    model,
    eval_loader,
    device,
)
