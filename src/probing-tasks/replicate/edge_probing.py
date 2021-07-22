import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import tqdm

torch.multiprocessing.set_start_method('spawn', force=True)
import transformers

from edge_probing_utils import (BertEdgeProbingSingleSpan,
                                BertEdgeProbingTwoSpan)

JiantData = Tuple[
    List[str],
    List[List[int]],
    List[List[int]],
    List[str]
    ]

@dataclass
class TrainConfig():

    """Class for carrying the parameters for training.

    Explanations for some variables:
        lr: learning rate, needs to be the same as the learning rate the optimizer was initialized
            with (default 0.0001).
        max_evals: maximum number of evaluations before stopping the training. If None, max_evals_wo_improvement
            is used to determine when to stop training (default None).
        max_evals_per_lr: maximum number of evaluations without improvement before the learning
            rate is halved (default 5).
        max_evals_wo_improvement: maximum number of evaluations without improvement before training
            is stopped (default 20).
        eval_interval: number of batches between two evaluations (default 100).
        dev: device to run on (default None).
    """

    train_data: data.DataLoader
    val_data: data.DataLoader
    model: transformers.PreTrainedModel
    optimizer: optim.Optimizer
    loss_func: nn.modules.loss._Loss
    lr: float = 0.0001
    max_evals: int = None
    max_evals_per_lr: int = 5
    max_evals_wo_improvement: int = 20
    eval_interval: int = 100
    dev: torch.device = None

@dataclass
class ProbeConfig():

    """Class for carrying the parameters for probing.

    Explanations for some variables:
        num_layers: list of layers to probe.
        labels_to_ids: dict assigning each target label to an id.
        lr: learning rate, needs to be the same as the learning rate the optimizer was initialized
            with (default 0.0001).
        max_evals: maximum number of evaluations before stopping the training. If None, max_evals_wo_improvement
            is used to determine when to stop training (default None).
        max_evals_per_lr: maximum number of evaluations without improvement before the learning
            rate is halved (default 5).
        max_evals_wo_improvement: maximum number of evaluations without improvement before training
            is stopped (default 20).
        eval_interval: number of batches between two evaluations (default 100).
        dev: device to run on (default None).
    """

    train_data: data.DataLoader
    val_data: data.DataLoader
    test_data: data.DataLoader
    model_name: str
    num_layers: List[int]
    loss_func: nn.modules.loss._Loss
    labels_to_ids: Dict[str, int]
    task_type: str
    lr: float = 0.0001
    max_evals: int = None
    max_evals_per_lr: int = 5
    max_evals_wo_improvement: int = 20
    eval_interval: int = 100
    dev: torch.device = None
    results_path: str = None


def train_single_span(config: TrainConfig) -> None:
    print("Training the model")
    lr: int = config.lr
    eval: int = 0
    counter: int = 0
    start_index: int = 1
    # Train until one of the stop conditions is True.
    while True:
        config.model.train()
        loop: tqdm.notebook.tqdm_notebook = tqdm.tqdm_notebook(config.train_data)
        # Run through one epoch.
        for i, (xb, span1s, targets) in enumerate(loop, start_index):
            config.optimizer.zero_grad()
            output = config.model(
                input_ids=xb.to(config.dev),
                span1s=span1s.to(config.dev)
                )
            batch_loss = config.loss_func(output, targets.to(config.dev))
            batch_loss.backward()
            nn.utils.clip_grad_norm_(config.model.parameters(), 5.0)
            config.optimizer.step()
            # Evaluate the model after each eval_interval.
            if i % config.eval_interval == 0:
                print(f"Training run {eval+1} finished")
                loss: float = eval_single_span(config.val_data, config.model, config.loss_func, dev=config.dev)
                print(f"Loss: {loss}")
                eval += 1
                # Check if the model has improved.
                if eval == 1:
                    min_loss: float = loss
                if loss < min_loss:
                    min_loss = loss
                    counter = 0
                else:
                    counter += 1
                # Check if training is finished.
                if config.max_evals is not None and eval >= config.max_evals:
                    break
                elif counter >= config.max_evals_wo_improvement:
                    break
                elif counter % config.max_evals_per_lr == 0 and counter > 0:
                    lr = lr/2
                    print(f"No improvement for {config.max_evals_per_lr} epochs, halving the learning rate to {lr}")
                    for g in config.optimizer.param_groups:
                        g['lr'] = lr
        # If inner loop did not break start another epoch.
        else:
            start_index = i % config.eval_interval
            continue
        # If inner loop did break.
        break
    print("Training is finished")

def train_two_span(config: TrainConfig) -> None:
    print("Training the model")
    lr = config.lr
    eval: int = 0
    counter: int = 0
    start_index: int = 1
    # Train until one of the stop conditions is True.
    while True:
        config.model.train()
        loop: tqdm.notebook.tqdm_notebook = tqdm.tqdm_notebook(config.train_data)
        # Run through one epoch.
        for i, (xb, span1s, span2s, targets) in enumerate(loop, start_index):
            config.optimizer.zero_grad()
            output = config.model(
                input_ids=xb.to(config.dev),
                span2s=span2s.to(config.dev),
                span1s=span1s.to(config.dev)
                )
            batch_loss = config.loss_func(output, targets.to(config.dev))
            batch_loss.backward()
            nn.utils.clip_grad_norm_(config.model.parameters(), 5.0)
            config.optimizer.step()
            # Evaluate the model after each eval_interval.
            if i % config.eval_interval == 0:
                print(f"Training run {eval+1} finished")
                loss: float = eval_two_span(config.val_data, config.model, config.loss_func, dev=config.dev)
                print(f"Loss: {loss}")
                eval += 1
                # Check if the model has improved.
                if eval == 1:
                    min_loss: float = loss
                if loss < min_loss:
                    min_loss = loss
                    counter = 0
                else:
                    counter += 1
                print(f"No improvement for {counter} evaluations")
                # Check if training is finished.
                if config.max_evals is not None and eval >= config.max_evals:
                    break
                elif counter >= config.max_evals_wo_improvement:
                    break
                elif counter % config.max_evals_per_lr == 0 and counter > 0:
                    lr = lr/2
                    print(f"No improvement for {config.max_evals_per_lr} epochs, halving the learning rate to {lr}")
                    for g in config.optimizer.param_groups:
                        g['lr'] = lr
        # If inner loop did not break start anther epoch.
        else:
            start_index = i % config.eval_interval
            continue
        # If inner loop did break.
        break
    print("Training is finished")

def train_single_span_old(
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
    epoch: int = 0
    best_epoch: int = 1
    counter: int = 0
    while counter < max_epochs:
        epoch += 1
        print(f"Epoch {epoch}")
        model.train()
        loop = tqdm.tqdm_notebook(train_data)
        for i, (xb, span1s, targets) in enumerate(loop):
            optimizer.zero_grad()
            output = model(
                input_ids=xb.to(dev),
                span1s=span1s.to(dev)
                )
            batch_loss = loss_func(output, targets.to(dev))
            batch_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            if i % 1000 == 0:
                loss: float = eval_two_span(val_data, model, loss_func, dev=dev)
                print(f"Loss: {loss}")
                if epoch == 1:
                    min_loss: float = loss
                    continue
                if loss < min_loss:
                    min_loss = loss
                    if save_path:
                        best_epoch = epoch
                        torch.save(model.state_dict(), f"{save_path}/{epoch}")
                else:
                    counter += 1
                print(f"No improvement for {counter} evaluations")
                if counter >= max_epochs_per_lr:
                    lr = lr/2
                    print(f"No improvement for {max_epochs_per_lr} epochs, halving the learning rate to {lr}")
                    for g in optimizer.param_groups:
                        g['lr'] = lr
    print(f"No improvement for {max_epochs} epochs, training is finished")
    if save_path:
        print(f"Reverting back to the best model from epoch {best_epoch} with loss {min_loss}")
        model.load_state_dict(torch.load(f"{save_path}/{best_epoch}"))

def train_two_span_old(
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
    epoch: int = 0
    best_epoch: int = 1
    counter: int = 0
    while counter < max_epochs:
        epoch += 1
        print(f"Epoch {epoch}")
        model.train()
        loop = tqdm.tqdm_notebook(train_data)
        for i, (xb, span1s, span2s, targets) in enumerate(loop):
            optimizer.zero_grad()
            output = model(
                input_ids=xb.to(dev),
                span1s=span1s.to(dev),
                span2s=span2s.to(dev)
                )
            batch_loss = loss_func(output, targets.to(dev))
            batch_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            if i % 1000 == 0:
                loss = eval_two_span(val_data, model, loss_func, dev=dev)
                print(f"Loss: {loss}")
                if epoch == 1:
                    min_loss: float = loss
                    continue
                if loss < min_loss:
                    min_loss = loss
                    if save_path:
                        torch.save(model.state_dict(), f"{save_path}/{epoch}")
                        best_epoch = epoch
                else:
                    counter += 1
                if counter >= max_epochs_per_lr:
                    lr = lr/2
                    print(f"No improvement for {max_epochs_per_lr} epochs, halving the learning rate to {lr}")
                    for g in optimizer.param_groups:
                        g['lr'] = lr
    print(f"No improvement for {max_epochs} epochs, training is finished")
    if save_path:
        print(f"Reverting back to the best model from epoch {best_epoch} with loss {min_loss}")
        model.load_state_dict(torch.load(f"{save_path}/{best_epoch}"))

def eval_single_span(val_data: data.DataLoader, model, loss_func, dev=None) -> float:
    print("Evaluating the model")
    model.eval()
    loop = tqdm.tqdm_notebook(val_data)
    with torch.no_grad():
        return sum(loss_func(model(xb.to(dev), span1s.to(dev)), targets.to(dev)).mean().item()
                   for xb, span1s, targets in loop) / len(loop)

def eval_two_span(val_data: data.DataLoader, model, loss_func, dev=None) -> float:
    print("Evaluating the model")
    model.eval()
    loop = tqdm.tqdm_notebook(val_data)
    with torch.no_grad():
        return sum(
            loss_func(model(
                input_ids=xb.to(dev),
                span1s=span1s.to(dev),
                span2s=span2s.to(dev)), targets.to(dev)).mean().item()
            for xb, span1s, span2s, targets in loop) / len(loop)

def test_single_span(test_data: data.DataLoader, model, loss_func, ids, dev=None) -> Tuple[float, float, float]:
    print("Testing the model")
    model.eval()
    loop = tqdm.tqdm_notebook(test_data)
    preds_sum = {id: 0.0 for id in ids}
    targets_sum = {id: 0.0 for id in ids}
    preds_correct_sum = {id: 0.0 for id in ids}
    loss = 0.0
    accuracy = 0.0
    with torch.no_grad():
        for xb, span1s, targets in loop:
            output = model(input_ids=xb.to(dev), span1s=span1s.to(dev))
            targets = targets.to(dev)
            loss += loss_func(output, targets).float().mean().item()
            preds = torch.argmax(output, dim=1)
            targets_max = torch.argmax(targets, dim=1)
            for id in ids:
                preds_sum[id] += torch.sum(preds == id).item()
                targets_sum[id] += torch.sum(targets_max == id).item()
                preds_correct_sum[id] += torch.sum((preds == id) * (targets_max == id)).item()
            accuracy += (preds == targets_max).float().mean().item()
    precision = [preds_correct_sum[id]/preds_sum[id] if preds_sum[id] != 0 else 0 for id in ids]
    recall = [preds_correct_sum[id]/targets_sum[id] if targets_sum[id] != 0 else 0 for id in ids]
    macro_precision = sum(precision) / len(ids)
    macro_recall = sum(recall) / len(ids)
    macro_f1_score = 2*(macro_precision*macro_recall) / (macro_precision + macro_recall)
    return loss / len(loop), accuracy / len(loop), macro_f1_score

def test_two_span(test_data, model, loss_func, ids, dev=None) -> Tuple[float, float, float]:
    print("Testing the model")
    model.eval()
    loop = tqdm.tqdm_notebook(test_data)
    preds_sum = {id: 0.0 for id in ids}
    targets_sum = {id: 0.0 for id in ids}
    preds_correct_sum = {id: 0.0 for id in ids}
    loss = 0.0
    accuracy = 0.0
    with torch.no_grad():
        for xb, span1s, span2s, targets in loop:
            output = model(input_ids=xb.to(dev), span1s=span1s.to(dev), span2s=span2s.to(dev))
            targets = targets.to(dev)
            loss += loss_func(output, targets).float().mean().item()
            preds = torch.argmax(output, dim=1)
            targets_max = torch.argmax(targets, dim=1)
            for id in ids:
                preds_sum[id] += torch.sum(preds == id).item()
                targets_sum[id] += torch.sum(targets_max == id).item()
                preds_correct_sum[id] += torch.sum((preds == id) * (targets_max == id)).item()
            accuracy += (preds == targets_max).float().mean().item()
    precision = [preds_correct_sum[id]/preds_sum[id] if preds_sum[id] != 0 else 0 for id in ids]
    recall = [preds_correct_sum[id]/targets_sum[id] if targets_sum[id] != 0 else 0 for id in ids]
    macro_precision = sum(precision) / len(ids)
    macro_recall = sum(recall) / len(ids)
    macro_f1_score = 2*(macro_precision*macro_recall) / (macro_precision + macro_recall)
    return loss / len(loop), accuracy / len(loop), macro_f1_score

def probing(config: ProbeConfig):
    """Probe a transformer model according to the edge probing concept.

    For each given layer initialize a model for probing on top of a given transformer model.
    Train and evaluate the model and return loss, accuracy, f1_score for each layer.

    Args:
        config: configuration specifying the run paramaters.
    Returns:
        dict: containing the results for each layer.
    """
    results = {}
    print(f"Probing model {config.model_name}")
    for layer in config.num_layers:
        print(f"Probing layer {layer} of {config.num_layers[-1]}")
        if config.task_type == "single_span":
            probing_model = BertEdgeProbingSingleSpan.from_pretrained(
                config.model_name,
                num_labels = len(config.labels_to_ids.keys()),
                num_hidden_layers=layer
                ).to(config.dev)
            optimizer = optim.Adam(probing_model.parameters(), lr=config.lr)
            train_single_span(TrainConfig(
                config.train_data,
                config.val_data,
                probing_model,
                optimizer,
                config.loss_func,
                lr=config.lr,
                max_evals=config.max_evals,
                max_evals_per_lr=config.max_evals_per_lr,
                max_evals_wo_improvement=config.max_evals_wo_improvement,
                eval_interval=config.eval_interval,
                dev=config.dev
                ))
            loss, accuracy, f1_score = test_single_span(
                config.test_data,
                probing_model,
                config.loss_func,
                config.labels_to_ids.values(),
                dev=config.dev)
        elif config.task_type == "two_span":
            probing_model = BertEdgeProbingTwoSpan.from_pretrained(
                config.model_name,
                num_labels = len(config.labels_to_ids.keys()),
                num_hidden_layers=layer
                ).to(config.dev)
            optimizer = optim.Adam(probing_model.parameters(), lr=config.lr)
            train_two_span(TrainConfig(
                config.train_data,
                config.val_data,
                probing_model,
                optimizer,
                config.loss_func,
                lr=config.lr,
                max_evals=config.max_evals,
                max_evals_per_lr=config.max_evals_per_lr,
                max_evals_wo_improvement=config.max_evals_wo_improvement,
                eval_interval=config.eval_interval,
                dev=config.dev
                ))
            loss, accuracy, f1_score = test_two_span(
                config.test_data,
                probing_model,
                config.loss_func,
                config.labels_to_ids.values(),
                dev=config.dev)
        else:
            print(f"{config.task_type} is not a valid task type")
            return None
        results[layer] = {"loss": loss, "accuracy": accuracy, "f1_score": f1_score}
        print(f"Test loss: {loss}, accuracy: {accuracy}, f1_score: {f1_score}")
        if config.results_path is not None:
            with open(f"{config.results_path}/results.json", "w") as f:
                json.dump(results, f)            
    return results

def read_jiant_dataset(input_path: str) -> JiantData:
    """Read a *.jsonl file in jiant format and return the relevant content."""
    print(f"Reading {input_path}")
    with open(input_path, "r") as f:
        lines = tqdm.tqdm_notebook(f.readlines())

    texts: List[str] = []
    span1s: List[List[int]] = []
    span2s: List[List[int]] = []
    labels: List[str] = []

    for line in lines:
        jiant_dict = json.loads(line)
        # In case there are no targets, save the text and append to it the next line.
        targets = jiant_dict["targets"]
        for target in targets:
            texts.append(jiant_dict["text"])
            span1s.append(target["span1"])
            if "span2" in target.keys():
                span2s.append(target["span2"])
            labels.append(target["label"])

    return texts, span1s, span2s, labels

def tokenize_jiant_dataset(
        tokenizer,
        texts: List[str],
        span1s: List[List[int]],
        span2s: List[List[int]],
        labels: List[str],
        labels_to_ids: Dict[str, int],
        max_seq_length: int=128,
        ) -> transformers.BatchEncoding:
    """Prepare a jiant dataset for the DataSet function.

    This function does all the heavy work, so that while training the model __getitem__ runs fast.
    """
    print("Tokenizing")
    encodings = tokenizer(texts, return_tensors="pt", max_length=128, truncation=True, padding="max_length")
    label_tensors: List[torch.Tensor] = []
    span1_masks: List[torch.Tensor] = []
    span2_masks: List[torch.Tensor] = []
    updated_input_ids: List[torch.Tensor] = []
    num_labels: int = len(labels_to_ids.keys())

    for i, tokens in enumerate(tqdm.tqdm_notebook(encodings.input_ids)):
        # Pad the sequence with zeros, if it's too short.
        target = torch.zeros(max_seq_length).int()
        target[:tokens.shape[0]] = tokens

        # Calculate the spans and check if they are valid
        # Otherwise continue to the next sample.
        
        # Differentiate between dataset with span2s and dataset without span2s. 
        if span2s[i]:
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
        elif span1s[i]:
            spans = [
                encodings.word_to_tokens(span1s[i][0]),
                encodings.word_to_tokens(span1s[i][1]),
                ]
            if None in spans:
                continue
            span1_start, _ = spans[0]
            _, span1_end = spans[1]
            span2_start, span2_end = (0, 0)
        # Calculate a mask tensor [x_0,...,x_max_sequence_length]
        # with x_i=1 if if span1[0]<=i<span1[1].
        span1_mask_start = (torch.arange(max_seq_length) >= span1_start)
        span1_mask_end = (torch.arange(max_seq_length) < span1_end)
        span1_mask = torch.minimum(span1_mask_start, span1_mask_end)
        span2_mask_start = (torch.arange(max_seq_length) >= span2_start)
        span2_mask_end = (torch.arange(max_seq_length) < span2_end)
        span2_mask = torch.minimum(span2_mask_start, span2_mask_end)

        label_tensor = torch.tensor(
            [1 if labels_to_ids[labels[i]]==k else 0 for k in range(num_labels)]
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
