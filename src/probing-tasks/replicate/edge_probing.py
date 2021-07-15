from edge_probing_utils import (
    BertEdgeProbingSingleSpan,
    BertEdgeProbingTwoSpan
    )

from transformers import BatchEncoding
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
torch.multiprocessing.set_start_method('spawn', force=True)

import matplotlib.pyplot as plt

import json
from tqdm import tqdm

from typing import List, Tuple, Dict

JiantData = Tuple[
    List[str],
    List[List[int]],
    List[List[int]],
    List[str]
    ]

def train_single_span(
        train_data,
        val_data,
        model,
        optimizer,
        loss_func,
        epochs: int,
        dev = None,
        save_path: str = None,
        ):
    print("Training the model")
    losses = []
    if save_path is not None and epochs > 1:
        save: bool = True
    else:
        save: bool = False
    for epoch in epochs:
        print(f"Epoch {epoch+1} of {epochs}")
        model.train()
        loop = tqdm(train_data)
        for xb, span1s, targets in loop:
            optimizer.zero_grad()
            output = model(
                input_ids=xb.to(dev),
                span1s=span1s.to(dev)
                )
            loss = loss_func(output, targets.to(dev))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        losses.append(eval_single_span(val_data, model, loss_func, dev=dev))
        if save:
            torch.save(model.state_dict(), f"{save_path}/{epoch}")
        print(f"Loss: {losses[epoch]}")
    epoch_min, loss_min = max(enumerate(losses), key=lambda x: x[1])
    if save:
        print(f"Reverting back to the best model from epoch {epoch_min+1} with loss {loss_min}")
        model.load_state_dict(torch.load(f"{save_path}/{epoch_min}"))

def train_two_span(
        train_data: DataLoader,
        val_data: DataLoader,
        model,
        optimizer,
        loss_func,
        epochs: int,
        dev = None,
        save_path: str = None,
        ):
    print("Training the model")
    losses = []
    if save_path is not None and epochs > 1:
        save: bool = True
    else:
        save: bool = False
    for epoch in range(epochs):
        print(f"Epoch {epoch+1} of {epochs}")
        model.train()
        loop = tqdm(train_data)
        for xb, span1s, span2s, targets in loop:
            optimizer.zero_grad()
            output = model(
                input_ids=xb.to(dev),
                span1s=span1s.to(dev),
                span2s=span2s.to(dev)
                )
            loss = loss_func(output, targets.to(dev))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        losses.append(eval_two_span(val_data, model, loss_func, dev=dev))
        if save:
            torch.save(model.state_dict(), f"{save_path}/{epoch}")
        print(f"Loss: {losses[epoch]}")
    epoch_min, loss_min = max(enumerate(losses), key=lambda x: x[1])
    if save:
        print(f"Reverting back to the best model from epoch {epoch_min+1} with loss {loss_min}")
        model.load_state_dict(torch.load(f"{save_path}/{epoch_min}"))

def eval_single_span(val_data, model, loss_func, dev=None):
    print("Evaluating the model")
    model.eval()
    loop = tqdm(val_data)
    with torch.no_grad():
        return sum(loss_func(model(xb.to(dev), span1s.to(dev)), targets).mean().item()
                   for xb, span1s, targets in loop) / len(loop)

def eval_two_span(val_data, model, loss_func, dev=None):
    print("Evaluating the model")
    model.eval()
    loop = tqdm(val_data)
    with torch.no_grad():
        return sum(
            loss_func(model(
                input_ids=xb.to(dev),
                span1s=span1s.to(dev),
                span2s=span2s.to(dev)), targets.to(dev)).mean().item()
            for xb, span1s, span2s, targets in loop) / len(loop)

def test_single_span(test_data, model, loss_func, ids, dev=None):
    print("Testing the model")
    model.eval()
    loop = tqdm(test_data)
    preds_sum = {id: 0.0 for id in ids}
    targets_sum = {id: 0.0 for id in ids}
    preds_correct_sum = {id: 0.0 for id in ids}
    loss = 0.0
    accuracy = 0.0
    with torch.no_grad():
        for xb, span1s, targets in loop:
            targets = targets.to(dev)
            output = model(input_ids=xb.to(dev), span1s=span1s.to(dev))
            preds = output > 0.5
            for id in ids:
                preds_sum[id] += torch.sum(preds == id).item()
                targets_sum[id] += torch.sum(targets == id).item()
                preds_correct_sum[id] += torch.sum((preds == id) * (targets == id)).item()
            loss += loss_func(output, targets).float().mean()
            accuracy += (preds == targets).float().mean()
    precision = [preds_correct_sum[id] / preds_sum[id] if preds_sum[id] != 0 else 0 for id in ids]
    recall = [preds_correct_sum[id] / targets_sum[id] if targets_sum[id] != 0 else 0 for id in ids]
    f1_score = sum(
        2*(precision[id]*recall[id])/(precision[id] + recall[id]) if (precision[id] + recall[id]) != 0
        else 0 for id in ids
        )
    return loss / len(loop), accuracy / len(loop), f1_score / len(ids)

def test_two_span(test_data, model, loss_func, ids, dev=None):
    print("Testing the model")
    model.eval()
    loop = tqdm(test_data)
    preds_sum = {id: 0.0 for id in ids}
    targets_sum = {id: 0.0 for id in ids}
    preds_correct_sum = {id: 0.0 for id in ids}
    loss = 0.0
    accuracy = 0.0
    with torch.no_grad():
        for xb, span1s, span2s, targets in loop:
            targets = targets.to(dev)
            output = model(input_ids=xb.to(dev), span1s=span1s.to(dev), span2s=span2s.to(dev))
            preds = output > 0.5
            for id in ids:
                preds_sum[id] += torch.sum(preds == id).item()
                targets_sum[id] += torch.sum(targets == id).item()
                preds_correct_sum[id] += torch.sum((preds == id) * (targets == id)).item()
            loss += loss_func(output, targets).float().mean()
            accuracy += (preds == targets).float().mean()
    precision = [preds_correct_sum[id] / preds_sum[id] if preds_sum[id] != 0 else 0 for id in ids]
    recall = [preds_correct_sum[id] / targets_sum[id] if targets_sum[id] != 0 else 0 for id in ids]
    f1_score = sum(
        2*(precision[id]*recall[id])/(precision[id] + recall[id]) if (precision[id] + recall[id]) != 0
        else 0 for id in ids
        )
    return loss / len(loop), accuracy / len(loop), f1_score / len(ids)

def probing(
        train_data: DataLoader,
        val_data: DataLoader,
        test_data: DataLoader,
        model_name: str,
        num_layers: List[int],
        loss_func,
        label_to_id,
        task_type: str,
        epochs: int=5,
        dev=None,
        ):
    results = {}
    print(f"Probing model {model_name}")
    for layer in num_layers:
        print(f"Probing layer {layer} of {num_layers[-1]}")
        if task_type == "single_span":
            probing_model = BertEdgeProbingSingleSpan.from_pretrained(
                model_name,
                num_labels = len(label_to_id.keys()),
                num_hidden_layers=layer
                ).to(dev)
            optimizer = optim.Adam(probing_model.parameters(), lr=0.0001)
            train_single_span(train_data, val_data, probing_model, optimizer, loss_func, epochs, dev=dev)
            loss, accuracy, f1_score = test_single_span(test_data, probing_model, loss_func, label_to_id.values(), dev=dev)
        elif task_type == "two_span":
            probing_model = BertEdgeProbingTwoSpan.from_pretrained(
                model_name,
                num_hidden_layers=layer
                ).to(dev)
            optimizer = optim.Adam(probing_model.parameters(), lr=0.0001)
            train_two_span(train_data, val_data, probing_model, optimizer, loss_func, epochs, dev=dev)
            loss, accuracy, f1_score = test_two_span(test_data, probing_model, loss_func, label_to_id.values(), dev=dev)
        else:
            print(f"{task_type} is not a valid task type")
            return None

        results[layer] = {"loss": loss, "accuracy": accuracy, "f1_score": f1_score}
        print(f"Test loss: {loss}, accuracy: {accuracy}, f1_score: {f1_score}")
    return results

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
        target = torch.zeros(max_seq_length).int()
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
        span1_mask_start = (torch.arange(max_seq_length) >= span1_start)
        span1_mask_end = (torch.arange(max_seq_length) < span1_end)
        span1_mask = torch.minimum(span1_mask_start, span1_mask_end)
        span2_mask_start = (torch.arange(max_seq_length) >= span2_start)
        span2_mask_end = (torch.arange(max_seq_length) < span2_end)
        span2_mask = torch.minimum(span2_mask_start, span2_mask_end)

        label_tensor = torch.tensor(
            [0 if label_to_id[labels[i]]==k else 1 for k in range(num_labels)]
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

def plot_task(results, task, model, linestyle, num_layers):
    y = [float(results[task][model][layer]['f1_macro']) for layer in num_layers]
    plt.plot(num_layers, y, linestyle, label=model)
    plt.legend()
    plt.suptitle(task)

def plot_task_from_file(input_path, model, task, linestyle, num_layers):
    with open(input_path, "r") as f:
        results = json.load(f)
    plot_task(results, task, model, linestyle, num_layers)
