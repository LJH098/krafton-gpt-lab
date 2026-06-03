# -*- coding: utf-8 -*-
"""GPT 사전 학습 유틸리티 과제 템플릿."""

import matplotlib.pyplot as plt
import torch

try:
    from .model import GPTModel
except ImportError:
    from model import GPTModel


def calc_loss_batch(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: GPTModel,
    device: torch.device,
) -> torch.Tensor:
    """TODO: 한 배치를 device로 옮긴 뒤 다음 토큰 예측 cross entropy loss를 계산합니다."""
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0,1), target_batch.flatten()
    )
    return loss
    raise NotImplementedError("calc_loss_batch를 구현하세요.")


def calc_loss_loader(
    data_loader,
    model: GPTModel,
    device: torch.device,
    num_batches: int | None = None,
) -> float:
    """TODO: data_loader의 평균 loss를 계산합니다. 검증에서는 torch.no_grad()를 사용하세요."""
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    if num_batches == 0:
        return float("nan")

    was_training = model.training
    model.eval()
    with torch.no_grad():
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i >= num_batches:
                break
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            total_loss += loss.item()

    if was_training:
        model.train()

    return total_loss / num_batches


def get_lr_with_warmup(base_lr: float, global_step: int, warmup_steps: int) -> float:
    """linear warmup으로 현재 step의 learning rate를 계산합니다."""
    if warmup_steps <= 0:
        return base_lr
    warmup_progress = min(1.0, (global_step + 1) / warmup_steps)
    return base_lr * warmup_progress


def apply_learning_rate_warmup(
    optimizer: torch.optim.Optimizer,
    global_step: int,
    warmup_steps: int,
    base_lrs: list[float] | None = None,
) -> list[float]:
    """optimizer param group에 warmup learning rate를 적용하고 현재 lr 목록을 반환합니다."""
    if base_lrs is None:
        base_lrs = [
            group.setdefault("initial_lr", group["lr"])
            for group in optimizer.param_groups
        ]

    current_lrs = []
    for group, base_lr in zip(optimizer.param_groups, base_lrs):
        lr = get_lr_with_warmup(base_lr, global_step, warmup_steps)
        group["lr"] = lr
        current_lrs.append(lr)
    return current_lrs


def save_checkpoint(
    model: GPTModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    path: str,
) -> None:
    """TODO: model/optimizer 상태, epoch, global_step을 torch.save로 저장합니다."""
    torch.save({
            "model_state_dict" : model.state_dict(),
            "optimizer_state_dict" : optimizer.state_dict(),
            "epoch" : epoch,
            "global_step" : global_step,
        },
        path
    )
    return
    raise NotImplementedError("save_checkpoint를 구현하세요.")


def load_checkpoint(
    model: GPTModel,
    optimizer: torch.optim.Optimizer | None,
    path: str,
    device: torch.device,
) -> tuple[int, int]:
    """TODO: torch.load로 checkpoint를 읽어 model/optimizer 상태를 복원합니다."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    model.train()
    return checkpoint["epoch"], checkpoint["global_step"]
    raise NotImplementedError("load_checkpoint를 구현하세요.")


def generate(
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    eos_id: int | None = None,
) -> torch.Tensor:
    """TODO: temperature와 top-k 샘플링을 지원하는 생성 함수를 구현합니다."""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        if top_k is not None and top_k > 0:
            top_k = min(top_k, logits.size(-1))
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1, None]
            logits = torch.where(
                logits < min_val,
                torch.full_like(logits, float("-inf")),
                logits
            )
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        if eos_id is not None and torch.all(idx_next == eos_id):
            break
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
    raise NotImplementedError("generate를 구현하세요.")


def generate_and_print_sample(
    model: GPTModel,
    tokenizer,
    device: torch.device,
    start_context: str,
    max_new_tokens: int = 50,
    context_size: int = 256,
    temperature: float = 0.8,
    top_k: int | None = 40,
) -> None:
    """TODO: start_context를 encode하고 generate 후 decode하여 출력합니다."""
    was_training = model.training
    model.eval()
    encoded = tokenizer.encode(start_context)
    idx = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)
    eos_id = tokenizer.get_eos_id() if hasattr(tokenizer, "get_eos_id") else None
    with torch.no_grad():
        token_ids = generate(
            model=model, idx=idx,
            max_new_tokens=max_new_tokens, context_size=context_size,
            temperature=temperature, top_k=top_k, eos_id=eos_id
        )
    decoded = tokenizer.decode(token_ids.squeeze(0).tolist())
    print(decoded)

    if was_training:
        model.train()
    return
    raise NotImplementedError("generate_and_print_sample을 구현하세요.")


def train_model(
    model: GPTModel,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    eval_freq: int,
    eval_iter: int,
    start_context: str,
    tokenizer,
    ckpt_freq: int | None = None,
    start_epoch: int = 0,
    global_step: int = 0,
    warmup_steps: int = 0,
) -> list[float]:
    """TODO: 사전 학습 루프를 구현하고 epoch별 train loss 리스트를 반환합니다."""
    model.to(device)
    train_losses = []
    context_size = model.config.get("context_length", 256)
    base_lrs = [
        group.setdefault("initial_lr", group["lr"])
        for group in optimizer.param_groups
    ]

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for input_batch, target_batch in train_loader:
            current_lrs = apply_learning_rate_warmup(
                optimizer, global_step, warmup_steps, base_lrs
            )
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            if eval_freq > 0 and global_step % eval_freq == 0:
                train_loss = calc_loss_loader(
                    train_loader, model, device, num_batches=eval_iter
                )
                val_loss = (
                    calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
                    if val_loader is not None
                    else float("nan")
                )
                print(
                    f"Ep {epoch + 1} (Step {global_step}): "
                    f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}, "
                    f"LR {current_lrs[0]:.2e}"
                )

            if ckpt_freq is not None and ckpt_freq > 0 and global_step % ckpt_freq == 0:
                save_checkpoint(
                    model,
                    optimizer,
                    epoch + 1,
                    global_step,
                    f"checkpoint_step_{global_step}.pt",
                )

        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else float("nan")
        train_losses.append(avg_epoch_loss)
        print(f"Epoch {epoch + 1} train loss: {avg_epoch_loss:.3f}")

        if start_context:
            sample_context_size = (
                model.config.get("context_length", 256)
                if hasattr(model, "config")
                else 256
            )
            generate_and_print_sample(
                model,
                tokenizer,
                device,
                start_context,
                context_size=sample_context_size,
            )

    return train_losses
    raise NotImplementedError("train_model을 구현하세요.")


def plot_losses(train_losses: list[float], val_losses: list[float] | None = None) -> None:
    """훈련/검증 손실 그래프를 그리는 제공 함수."""
    plt.plot(train_losses, label="Train")
    if val_losses is not None:
        plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training / Validation Loss")
    plt.show()
