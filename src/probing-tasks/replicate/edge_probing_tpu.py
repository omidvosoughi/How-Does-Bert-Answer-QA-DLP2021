from dataclasses import dataclass
from typing import List, Tuple, Dict

from tqdm import tqdm_notebook as tqdm

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
torch.multiprocessing.set_start_method('spawn', force=True)

import torch_xla
import torch_xla.core.xla_model as xm

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

JiantData = Tuple[
    List[str],
    List[List[int]],
    List[List[int]],
    List[str]
    ]

def train_single_span(config):
    print("Training the model")
    lr = config.lr
    eval: int = 0
    counter: int = 0
    start_index: int = 1
    while True:
        config.model.train()
        loop = tqdm(config.train_data)
        for i, (xb, span1s, targets) in enumerate(loop, start_index):
            config.optimizer.zero_grad()
            output = config.model(
                input_ids=xb.to(config.dev),
                span1s=span1s.to(config.dev)
                )
            batch_loss = config.loss_func(output, targets.to(config.dev))
            batch_loss.backward()
            nn.utils.clip_grad_norm_(config.model.parameters(), 5.0)
            xm.optimizer_step(config.optimizer)
            xm.mark_step()
            if i % config.eval_interval == 0:
                loss = eval_single_span(config.val_data, config.model, config.loss_func, dev=config.dev)
                print(f"Loss: {loss}")
                eval += 1
                # Check if the model has improved.
                if eval == 1:
                    min_loss: float = loss
                    continue
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
                elif counter >= config.max_evals_per_lr:
                    lr = lr/2
                    print(f"No improvement for {config.max_evals_per_lr} epochs, halving the learning rate to {lr}")
                    for g in config.optimizer.param_groups:
                        g['lr'] = lr
                    counter = 0
        # If inner loop did not break.
        else:
            start_index = i % config.eval_interval
            continue
        # If inner loop did break.
        break
    print("Training is finished")

def train_two_span(config):
    print("Training the model")
    lr = config.lr
    eval: int = 0
    counter: int = 0
    start_index: int = 1
    while True:
        config.model.train()
        loop = tqdm(config.train_data)
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
            xm.optimizer_step(config.optimizer)
            xm.mark_step()
            if i % config.eval_interval == 0:
                loss = eval_two_span(config.val_data, config.model, config.loss_func, dev=config.dev)
                print(f"Loss: {loss}")
                eval += 1
                # Check if the model has improved.
                if eval == 1:
                    min_loss: float = loss
                    continue
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
                elif counter >= config.max_evals_per_lr:
                    lr = lr/2
                    print(f"No improvement for {config.max_evals_per_lr} epochs, halving the learning rate to {lr}")
                    for g in config.optimizer.param_groups:
                        g['lr'] = lr
                    counter = 0
        # If inner loop did not break.
        else:
            start_index = i % config.eval_interval
            continue
        # If inner loop did break.
        break
    print("Training is finished")

def probing(config) -> Dict[int, Dict[str, float]]:
    """Probe a transformer model according to the edge probing introduced by van Aken.
    
    For each given layer initialize a model for probing on top of a given transformer model.
    Train and evaluate the model and return loss, accuracy, f1_score for each layer.
    
    Args (in ProbingConfig):
    train_data --
    val_data: DataLoader
    test_data: DataLoader
    model_name: str
    num_layers: List[int]
    loss_func: nn.modules.loss._Loss
    labels_to_ids: Dict[str, int]
    task_type: str
    
    Keyword arguments (in ProbingConfig):
    lr: learning rate, needs to be the same as the learning rate the optimizer was initialized
          with (default 0.0001)
    max_evals: maximum number of evaluations before stopping the training. If None, max_evals_wo_improvement
                 is used to determine when to stop training (default None)
    max_evals_per_lr: maximum number of evaluations without improvement before the learning
                        rate is halved (default 5)
    max_evals_wo_improvement: maximum number of evaluations without improvement before training
                                is stopped (default 20)
    eval_interval: number of batches between two evaluations (default 100)
    dev: device on which the model runs (default None)
    
    Returns:
        Dict: 
    
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
    return results
