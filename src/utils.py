import json
import platform
from dataclasses import dataclass, field
from typing import List

import evaluate
import numpy as np
import torch
from tqdm import tqdm


def get_device():
    if platform.platform().lower().startswith("mac"):  # macOS
        return "mps" if torch.backends.mps.is_available() else "cpu"
    else:  # Linux, Windows
        return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Config:
    DATA_BASE: str = ""
    TRAIN_RAW: str = ""
    TRAIN_DATA: str = ""
    VAL_RAW: str = ""
    VAL_DATA: str = ""
    TRAIN_SOURCE: List[str] = field(default_factory=list)
    VAL_SOURCE: List[str] = field(default_factory=list)
    TRAIN_TARGET: List[str] = field(default_factory=list)
    VAL_TARGET: List[str] = field(default_factory=list)


def load_config(config_file="config.json"):
    with open(config_file, "r") as f:
        data = json.load(f)

    return Config(
        DATA_BASE=data["DATA_BASE"],
        TRAIN_RAW=data["TRAIN_RAW"],
        TRAIN_DATA=data["TRAIN_DATA"],
        VAL_RAW=data["VAL_RAW"],
        VAL_DATA=data["VAL_DATA"],
        TRAIN_SOURCE=data["TRAIN_SOURCE"],
        VAL_SOURCE=data["VAL_SOURCE"],
        TRAIN_TARGET=data["TRAIN_TARGET"],
        VAL_TARGET=data["VAL_TARGET"],
    )


def train_model(
    model,
    train_loader,
    optimizer,
    criterion,
    device,
    epochs,
    source_test,
    reference,
    translator,
):
    metric = evaluate.load("bleu", cache_dir="./")
    train_loss = []
    N = len(train_loader.dataset)
    for epoch in range(epochs):
        pbar = tqdm(train_loader, unit="batch", desc=f"Epoch {epoch + 1}/{epochs}")
        run_loss = 0
		model.train()
        for source, target in pbar:
            source = source.to(device)
            target = target.to(device)

            output = model(source, target)
            output = output.reshape(-1, output.shape[2])
            target = target.permute(1, 0).reshape(-1)

            optimizer.zero_grad()
            loss = criterion(output, target)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            run_loss += loss.item() * source.size(0) 
            pbar.set_postfix(loss=f"{run_loss/N:.3f}")
        train_loss.append(run_loss / N)
        predicted = translator.translate_sentence(source_test)
        bleu = sentence_bleu([predicted], [reference], metric)
        print(f"Predicted: {predicted}")
        print(f"BLEU Score: {bleu}")
    return train_loss


def sentence_bleu(prediction, reference, metric=evaluate.load("bleu", cache_dir="./")):
    blues = [
        np.round(
            metric.compute(predictions=prediction, references=reference, max_order=i)[
                "bleu"
            ],
            3,
        )
        for i in range(1, 5)
    ]
    return blues


def corpus_bleu(predictions, references):
    metric = evaluate.load("bleu")
    for i in range(1, 5):
        print(f"BLEU{-i}".center(80))
        print("-----" * 18)
        blues = metric.compute(
            predictions=predictions, references=references, max_order=i
        )
        for key, val in blues.items():
            print(f"{key:<20}: {val}")
        print("*****" * 18)
