from edge_probing_utils import (
    BertEdgeProbingSingleSpan,
    BertEdgeProbingTwoSpan,
    )
from edge_probing import (
    eval_single_span,
    eval_two_span,
    test_two_span,
    test_two_span,
    )

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
torch.multiprocessing.set_start_method('spawn', force=True)

import torch_xla
import torch_xla.core.xla_model as xm

from tqdm import tqdm

from typing import List, Tuple

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
        save_path: str = None,
        dev = None,
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
            xm.optimizer_step(optimizer)
            xm.mark_step()
        losses.append(eval_single_span(val_data, model, loss_func, dev=dev))
        if save:
            torch.save(model.state_dict(), f"{save_path}/{epoch}")
        print(f"Loss: {losses[epoch]}")
    epoch_min, loss_min = max(enumerate(losses), key=lambda x: x[1])
    if save:
        print(f"Reverting back to the best model from epoch {epoch_min} with loss {loss_min}")
        model.load_state_dict(torch.load(f"{save_path}/{epoch_min}"))

def train_two_span(
        train_data: DataLoader,
        val_data: DataLoader,
        model,
        optimizer,
        loss_func,
        epochs: int,
        save_path: str = None,
        dev = None,
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
            xm.optimizer_step(optimizer)
            xm.mark_step()
        losses.append(eval_two_span(val_data, model, loss_func, dev=dev))
        if save:
            torch.save(model.state_dict(), f"{save_path}/{epoch}")
        print(f"Loss: {losses[epoch]}")
    epoch_min, loss_min = max(enumerate(losses), key=lambda x: x[1])
    if save:
        print(f"Reverting back to the best model from epoch {epoch_min} with loss {loss_min}")
        model.load_state_dict(torch.load(f"{save_path}/{epoch_min}"))

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
                num_hidden_layers=layer
                ).to(dev)
            optimizer = optim.Adam(probing_model.parameters(), lr=0.0001)
            train_single_span(train_data, val_data, probing_model, optimizer, loss_func, epochs, dev=dev)
            loss, accuracy, f1_score = test_single_span(test_data, probing_model, loss_func, label_to_id.values())
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
