from edge_probing_utils import (
    BertEdgeProbingSingleSpan,
    BertEdgeProbingTwoSpan
    )
from transformers import BatchEncoding
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
torch.multiprocessing.set_start_method('spawn', force=True)

import json
from tqdm import tqdm

from typing import List, Tuple, Dict

JiantData = Tuple[
    List[str],
    List[List[int]],
    List[List[int]],
    List[str]
    ]

def train_single_span(train_data, val_data, model, optimizer, loss_function, epochs: int):
    print("Training the model")
    for epoch in epochs:
        print(f"Epoch {epoch+1} of {epochs}")
        model.train()
        loop = tqdm(train_data)
        for xb, span1, label in loop:
            optimizer.zero_grad()
            output = model(
                input_ids=xb,
                span1s=span1
                )
            loss = loss_function(output, label)
            loss.backward()
            optimizer.step()
        loss: float = eval_single_span(val_data, model, loss_func)
        print(f"Loss: {loss}")


def train_two_span(train_data: DataLoader, val_data: DataLoader, model, optimizer, loss_func, epochs: int):
    print("Training the model")
    for epoch in range(epochs):
        print(f"Epoch {epoch+1} of {epochs}")
        model.train()
        loop = tqdm(train_data)
        for xb, span1, span2, label in loop:
            optimizer.zero_grad()
            output = model(
                input_ids=xb,
                span1s=span1,
                span2s=span2
                )
            loss = loss_func(output, label)
            loss.backward()
            optimizer.step()
        loss: float = eval_two_span(val_data, model, loss_func)
        print(f"Loss: {loss}")

def eval_single_span(val_data, model, loss_func):
    print("Evaluating the model")
    model.eval()
    loop = tqdm(val_data)
    with torch.no_grad():
        return sum(loss_func(model(xb, span1), label).mean()
                   for xb, span1, label in loop) / len(loop)

def eval_two_span(val_data, model, loss_func):
    print("Evaluating the model")
    model.eval()
    loop = tqdm(val_data)
    with torch.no_grad():
        return sum(
            loss_func(model(
                input_ids=xb,
                span1s=span1,
                span2s=span2), label).mean()
            for xb, span1, span2, label in loop) / len(loop)

def probing(
        train_data: DataLoader,
        val_data: DataLoader,
        model_name: str,
        num_layers: List[int],
        loss_func,
        task_type: str,
        epochs: int=5,
        device=None
        ):
    for layer in num_layers:
        print(f"Probing layer {layer} of {num_layers[-1]}")
        if task_type == "single_span":
            probing_model = BertEdgeProbingSingleSpan.from_pretrained(
                model_name,
                num_hidden_layers=layer
                ).to(device)
            optimizer = optim.Adam(probing_model.parameters())
            train_single_span(train_data, val_data, probing_model, optimizer, loss_func, epochs)
        elif task_type == "two_span":
            probing_model = BertEdgeProbingTwoSpan.from_pretrained(
                model_name,
                num_hidden_layers=layer
                ).to(device)
            optimizer = optim.Adam(probing_model.parameters())
            train_two_span(train_data, val_data, probing_model, optimizer, loss_func, epochs)
        else:
            print(f"{task_type} is not a valid task type")


def read_jiant_dataset(input_path: str) -> JiantData:
    """Read a *.jsonl file in jiant format and return the relevant content."""
    print(f"Reading {input_path}")
    with open(input_path, "r") as f:
        lines = tqdm(f.readlines())

    texts: List[str] = []
    span1s: List[List[int]] = []
    span2s: List[List[int]] = []
    labels: List[str] = []

    text_save: str = ""

    for line in lines:
        jiant_dict = json.loads(line)
        # In case there are no targets, save the text and append to it the next line.
        text_save += jiant_dict["text"]
        targets = jiant_dict["targets"]
        if not targets:
            continue
        for target in targets:
            texts.append(text_save)
            span1s.append(target["span1"])
            span2s.append(target["span2"])
            labels.append(target["label"])
        text_save = ""


    return texts, span1s, span2s, labels

def tokenize_jiant_dataset(
        tokenizer,
        texts: List[str],
        span1s: List[List[int]],
        span2s: List[List[int]],
        labels: List[str],
        label_to_id: Dict[str, int],
        max_seq_length: int=128,
        device=None
        ) -> BatchEncoding:
    """Prepare a jiant dataset for the DataSet function.

    This function does all the heavy work, so that while training the model __getitem__ runs fast.
    """
    print("Tokenizing")
    encodings = tokenizer(texts, return_tensors="pt", max_length=128, truncation=True, padding="max_length")
    label_tensors: List[torch.Tensor] = []
    span1_masks: List[torch.Tensor] = []
    span2_masks: List[torch.Tensor] = []
    updated_input_ids: List[torch.Tensor] = []
    num_labels: int = len(label_to_id.keys())

    for i, tokens in enumerate(tqdm(encodings.input_ids)):
        # Pad the sequence with zeros, if it's too short.
        target = torch.zeros(max_seq_length, device=device).int()
        target[:tokens.shape[0]] = tokens

        # Calculate the spans and check if they are valid
        # Otherwise continue to the next sample.
        spans = [
            encodings.word_to_tokens(span1s[i][0]),
            encodings.word_to_tokens(span1s[i][1]),
            encodings.word_to_tokens(span2s[i][0]),
            encodings.word_to_tokens(span2s[i][1])
            ]
        if None in spans:
            continue
        span1_start, _ = spans[0]
        _, span1_end = spans[1]
        span2_start, _ = spans[2]
        _, span2_end = spans[3]
        # Calculate a mask tensor [x_0,...,x_max_sequence_length]
        # with x_i=1 if if span1[0]<=i<span1[1].
        span1_mask_start = (torch.arange(max_seq_length, device=device) >= span1_start)
        span1_mask_end = (torch.arange(max_seq_length, device=device) < span1_end)
        span1_mask = torch.minimum(span1_mask_start, span1_mask_end).to(device)
        span2_mask_start = (torch.arange(max_seq_length, device=device) >= span2_start)
        span2_mask_end = (torch.arange(max_seq_length, device=device) < span2_end)
        span2_mask = torch.minimum(span2_mask_start, span2_mask_end).to(device)

        label_tensor = torch.tensor(
            [0 if label_to_id[labels[i]]==k else 1 for k in range(num_labels)],
            device=device
            ).float()

        updated_input_ids.append(target)
        span1_masks.append(span1_mask)
        span2_masks.append(span2_mask)
        label_tensors.append(label_tensor)

    encodings.update({
        "input_ids": updated_input_ids,
        "span1_masks": span1_masks,
        "span2_masks": span2_masks,
        "labels": label_tensors
        })
    return encodings
