import json
import platform
from dataclasses import dataclass, field
from typing import List

import torch
from tqdm import tqdm
from torcheval.metrics.functional import bleu_score


def get_device():
    if platform.platform().lower().startswith('mac'):  # macOS
        return "mps" if torch.backends.mps.is_available() else "cpu"
    else: # Linux, Windows
        return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Config:
    DATA_BASE: str = ""
    TRAIN_RAW: str = ""
    TRAIN_DATA: str = ""
    VAL_RAW: str = ""
    VAL_DATA: str = ""
    TRAIN_AFRIKAANS: List[str] = field(default_factory=list)
    TRAIN_ENGLISH: List[str] = field(default_factory=list)
    VAL_AFRIKAANS: List[str] = field(default_factory=list)
    VAL_ENGLISH: List[str] = field(default_factory=list)


def load_config(config_file="config.json"):
    with open(config_file, "r") as f:
        data = json.load(f)

    return Config(
        DATA_BASE=data["DATA_BASE"],
        TRAIN_RAW=data["TRAIN_RAW"],
        TRAIN_DATA=data["TRAIN_DATA"],
        VAL_RAW=data["VAL_RAW"],
        VAL_DATA=data["VAL_DATA"],
        TRAIN_AFRIKAANS=data["TRAIN_AFRIKAANS"],
        TRAIN_ENGLISH=data["TRAIN_ENGLISH"],
        VAL_AFRIKAANS=data["VAL_AFRIKAANS"],
        VAL_ENGLISH=data["VAL_ENGLISH"],
    )


def train_model(model, train_loader, optimizer, criterion, device, epochs, source_test, target_test, source_lang, target_lang):
    N = len(train_loader.dataset)
    for epoch in range(epochs):
        pbar = tqdm(train_loader, unit="batch", desc=f"Epoch {epoch + 1}/{epochs}")
        run_loss = 0
        for source, target in pbar:
            source = source.to(device)
            target = target.to(device)

            output= model(source, target)
            output = output.reshape(-1, output.shape[2])
            target = target.permute(1, 0).reshape(-1)

            optimizer.zero_grad()
            loss = criterion(output, target)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            run_loss += loss.item() * source.size(0)
            pbar.set_postfix(loss=f"{run_loss / N:.3f}")

        predicted = translate_sentece(model, source_test, source_lang, target_lang, device)
        bleu = torch_bleu_score([predicted], [target_test])
        print(f"Predicted: {predicted}")
        print(f"Reference: {target_test}")
        print(f"BLEU Score: {bleu.item()}")


def translate_sentece(model, sentence, source_lang, target_lang, device):
    text = [source_lang.stoi[word] for word in sentence.strip().split()]
    text = torch.tensor(text, dtype=torch.long).unsqueeze(1).to(device)
    translated = model.translate(text)
    return " ".join([target_lang.itos[idx] for idx in translated])


def torch_bleu_score(candidat, reference, device=None):
    n_gram = min(len(candidat[0].split()), len(reference[0].split()), 4)
    return bleu_score(candidat, reference, n_gram, device=device)
