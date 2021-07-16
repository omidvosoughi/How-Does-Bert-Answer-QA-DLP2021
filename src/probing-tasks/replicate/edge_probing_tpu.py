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
        lr: float,
        max_epochs_per_lr: int=5,
        max_epochs: int=20,
        dev = None,
        save_path: str = None,
        ):
    print("Training the model")
    losses = []
    epoch = 1
    counter = 0
    while counter < max_epochs:
        print(f"Epoch {epoch}")
        epoch += 1
        model.train()
        loop = tqdm(train_data)
        for i, (xb, span1s, span2s, targets) in enumerate(loop):
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
        print(f"Loss: {losses[-1]}")
        if len(losses) > 1 and losses[-1] >= losses[-2]:
            counter += 1
        else:
            if save_path:
                torch.save(model.state_dict(), f"{save_path}/{epoch}")
            continue
        if counter >= max_epochs_per_lr:
            lr = lr/2
            print(f"No improvement for [{max_epochs_per_lr} epochs, halving the learning rate to {lr}")
            for g in optimizer.param_groups:
                g['lr'] = lr
    print(f"No improvement for 2{max_epochs} epochs, training is finished")
    if save_path:
        epoch_min, loss_min = max(enumerate(losses), key=lambda x: x[1])
        print(f"Reverting back to the best model from epoch {epoch_min+1} with loss {loss_min}")
        model.load_state_dict(torch.load(f"{save_path}/{epoch_min}"))

def train_two_span(
        train_data,
        val_data,
        model,
        optimizer,
        loss_func,
        lr: float,
        max_epochs_per_lr: int=5,
        max_epochs: int=20,
        dev = None,
        save_path: str = None,
        ):
    print("Training the model")
    losses = []
    epoch = 1
    counter = 0
    while counter < max_epochs:
        print(f"Epoch {epoch}")
        epoch += 1
        model.train()
        loop = tqdm(train_data)
        for i, (xb, span1s, span2s, targets) in enumerate(loop):
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
            if i >= 999:
                break
        losses.append(eval_two_span(val_data, model, loss_func, dev=dev))
        print(f"Loss: {losses[-1]}")
        if len(losses) > 1 and losses[-1] >= losses[-2]:
            counter += 1
        else:
            if save_path:
                torch.save(model.state_dict(), f"{save_path}/{epoch}")
            continue
        if counter >= max_epochs_per_lr:
            lr = lr/2
            print(f"No improvement for {max_epochs_per_lr} epochs, halving the learning rate to {lr}")
            for g in optimizer.param_groups:
                g['lr'] = lr
    print(f"No improvement for {max_epochs} epochs, training is finished")
    if save_path:
        epoch_min, loss_min = max(enumerate(losses), key=lambda x: x[1])
        print(f"Reverting back to the best model from epoch {epoch_min+1} with loss {loss_min}")
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
            train_single_span(train_data, val_data, probing_model, optimizer, loss_func, lr=0.0001, dev=dev)
            loss, accuracy, f1_score = test_single_span(test_data, probing_model, loss_func, label_to_id.values())
        elif task_type == "two_span":
            probing_model = BertEdgeProbingTwoSpan.from_pretrained(
                model_name,
                num_hidden_layers=layer
                ).to(dev)
            optimizer = optim.Adam(probing_model.parameters(), lr=0.0001)
            train_two_span(train_data, val_data, probing_model, optimizer, loss_func, lr=0.0001, dev=dev)
            loss, accuracy, f1_score = test_two_span(test_data, probing_model, loss_func, label_to_id.values(), dev=dev)
        else:
            print(f"{task_type} is not a valid task type")
            return None

        results[layer] = {"loss": loss, "accuracy": accuracy, "f1_score": f1_score}
        print(f"Test loss: {loss}, accuracy: {accuracy}, f1_score: {f1_score}")
    return results
