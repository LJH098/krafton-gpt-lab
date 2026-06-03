# -*- coding: utf-8 -*-
"""NSMC 감성 분류 미세 조정 유틸리티."""

import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset

try:
    from .model import GPTModel
except ImportError:
    from model import GPTModel


def _read_nsmc_tsv(path: str | Path) -> list[dict]:
    """NSMC TSV 파일을 [{"text": ..., "label": ...}] 형식으로 읽습니다."""
    rows = []
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        try:
            doc_idx = header.index("document")
            label_idx = header.index("label")
        except ValueError as exc:
            raise ValueError(f"{path} must contain document and label columns") from exc

        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(doc_idx, label_idx):
                continue

            if len(parts) > len(header):
                document = "\t".join(parts[doc_idx:-1])
                label_text = parts[-1]
            else:
                document = parts[doc_idx]
                label_text = parts[label_idx]

            document = document.strip()
            if not document:
                continue

            try:
                label = int(label_text)
            except ValueError:
                continue
            if label not in (0, 1):
                continue

            rows.append({"text": document, "label": label})

    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_sentiment_dataset(
    train_tsv_path: str | Path,
    test_tsv_path: str | Path | None = None,
    val_ratio: float = 0.08,
    seed: int = 42,
    output_dir: str | Path | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    NSMC TSV를 읽어 train/validation/test 감성 분류 데이터를 만듭니다.

    반환 형식:
        [{"text": "리뷰", "label": 0 또는 1}, ...]
    """
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must satisfy 0 <= val_ratio < 1")

    train_rows = _read_nsmc_tsv(train_tsv_path)
    random.Random(seed).shuffle(train_rows)

    val_size = int(len(train_rows) * val_ratio)
    if val_ratio > 0 and len(train_rows) > 1:
        val_size = max(1, val_size)
    val_size = min(val_size, max(len(train_rows) - 1, 0))

    val_data = train_rows[:val_size]
    train_data = train_rows[val_size:]
    test_data = _read_nsmc_tsv(test_tsv_path) if test_tsv_path is not None else []

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        _write_jsonl(output_path / "nsmc_sentiment_train.jsonl", train_data)
        _write_jsonl(output_path / "nsmc_sentiment_val.jsonl", val_data)
        _write_jsonl(output_path / "nsmc_sentiment_test.jsonl", test_data)

    return train_data, val_data, test_data


class ReviewSentimentDataset(Dataset):
    """감성 분류용 Dataset. 리뷰 하나와 label 하나를 반환합니다."""

    def __init__(
        self,
        data: list[dict],
        tokenizer,
        max_length: int = 128,
        pad_id: int | None = None,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_id = tokenizer.get_pad_id() if pad_id is None else pad_id
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """text를 encode하고 max_length까지 자르거나 padding한 뒤 label과 함께 반환합니다."""
        item = self.data[idx]
        text = str(item["text"])
        label = int(item["label"])

        try:
            token_ids = self.tokenizer.encode(text, add_bos_eos=True)
        except TypeError:
            token_ids = self.tokenizer.encode(text)

        if len(token_ids) > self.max_length:
            token_ids = token_ids[: self.max_length]
            if hasattr(self.tokenizer, "get_eos_id"):
                token_ids[-1] = self.tokenizer.get_eos_id()
        else:
            token_ids = token_ids + [self.pad_id] * (self.max_length - len(token_ids))

        return torch.tensor(token_ids, dtype=torch.long), label


class GPTForSequenceClassification(nn.Module):
    """
    GPT backbone 위에 감성 분류용 Linear head를 붙인 모델.

    주의: LM head는 다음 토큰 예측용입니다. 감성 분류는 hidden state 위에 별도 classifier를 붙입니다.
    """

    def __init__(
        self,
        gpt_model: GPTModel,
        num_labels: int = 2,
        drop_rate: float = 0.1,
    ):
        super().__init__()
        self.gpt = gpt_model
        self.num_labels = num_labels
        self.pad_id = gpt_model.config.get("pad_id", 0)
        self.dropout = nn.Dropout(drop_rate)
        self.classifier = nn.Linear(gpt_model.config["emb_dim"], num_labels)

    def _hidden_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        """GPTModel의 LM head 직전 hidden state를 계산합니다."""
        _, seq_len = input_ids.shape
        context_length = self.gpt.config["context_length"]
        if seq_len > context_length:
            raise ValueError(
                f"input sequence length {seq_len} exceeds context_length {context_length}"
            )

        token_embeddings = self.gpt.token_embedding(input_ids)
        positions = torch.arange(seq_len, device=input_ids.device)
        position_embeddings = self.gpt.position_embedding(positions)
        x = token_embeddings + position_embeddings
        x = self.gpt.drop_embedding(x)
        x = self.gpt.transformer_blocks(x)
        return self.gpt.final_layernorm(x)

    def _pool_last_non_pad(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """각 샘플에서 마지막 non-pad 토큰의 hidden state를 대표 벡터로 사용합니다."""
        batch_size, seq_len = input_ids.shape
        non_pad = input_ids.ne(self.pad_id)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        last_indices = torch.where(non_pad, positions, torch.zeros_like(positions)).max(dim=1).values
        batch_indices = torch.arange(batch_size, device=input_ids.device)
        return hidden_states[batch_indices, last_indices]

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        GPT hidden state에서 문장 대표 벡터를 뽑아 분류 logits를 만듭니다.

        labels가 있으면 (loss, logits), 없으면 logits를 반환합니다.
        """
        hidden_states = self._hidden_states(input_ids)
        pooled = self._pool_last_non_pad(hidden_states, input_ids)
        logits = self.classifier(self.dropout(pooled))

        if labels is None:
            return logits

        labels = labels.to(input_ids.device).long().view(-1)
        loss = nn.functional.cross_entropy(logits, labels)
        return loss, logits


def train_epoch_sentiment(
    model: GPTForSequenceClassification,
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """감성 분류 모델을 1 epoch 훈련하고 (평균 loss, accuracy)를 반환합니다."""
    model.to(device)
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for input_ids, labels in train_loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device).long().view(-1)

        optimizer.zero_grad(set_to_none=True)
        loss, logits = model(input_ids, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=-1) == labels).sum().item()
        total_examples += batch_size

    if total_examples == 0:
        return float("nan"), float("nan")
    return total_loss / total_examples, total_correct / total_examples


def evaluate_sentiment(
    model: GPTForSequenceClassification,
    data_loader,
    device: torch.device,
) -> tuple[float, float]:
    """감성 분류 모델을 평가하고 (평균 loss, accuracy)를 반환합니다."""
    model.to(device)
    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for input_ids, labels in data_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device).long().view(-1)

            loss, logits = model(input_ids, labels)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=-1) == labels).sum().item()
            total_examples += batch_size

    if was_training:
        model.train()

    if total_examples == 0:
        return float("nan"), float("nan")
    return total_loss / total_examples, total_correct / total_examples
