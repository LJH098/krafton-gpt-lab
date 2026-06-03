# -*- coding: utf-8 -*-
"""GPT 사전 학습용 Dataset/DataLoader 과제 템플릿."""

import torch
from torch.utils.data import DataLoader, Dataset


def encode_lm_corpus(
    text: str,
    tokenizer,
    add_bos_eos_per_line: bool = False,
    skip_empty_lines: bool = True,
) -> list[int]:
    """LM corpus를 token ID 리스트로 변환합니다."""
    if not add_bos_eos_per_line:
        return tokenizer.encode(text)

    token_ids = []
    for line in text.splitlines():
        if skip_empty_lines and not line:
            continue
        token_ids.extend(tokenizer.encode(line, add_bos_eos=True))
    return token_ids


class GPTDataset(Dataset):
    """
    token ID 리스트를 다음 토큰 예측용 input/target 쌍으로 자릅니다.

    예: token_ids=[10, 11, 12, 13], context_length=3
    - input:  [10, 11, 12]
    - target: [11, 12, 13]
    """

    def __init__(
        self,
        token_ids: list[int],
        context_length: int,
        stride: int | None = None,
    ):
        self.token_ids = token_ids
        self.context_length = context_length
        self.stride = stride if stride is not None else context_length
        # TODO: 만들 수 있는 학습 샘플 개수를 self._length에 저장하세요.
        self._length = max((len(self.token_ids) - self.context_length-1) // self.stride +1, 0)
        #raise NotImplementedError("GPTDataset.__init__에서 self._length를 구현하세요.")

    def __len__(self) -> int:
        """TODO: 전체 샘플 개수를 반환합니다."""
        return self._length
        raise NotImplementedError("GPTDataset.__len__을 구현하세요.")

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        TODO: idx번째 input_ids와 target_ids를 LongTensor로 반환합니다.

        Returns:
            input_ids: (context_length,)
            target_ids: (context_length,)
        """
        input_ids = self.token_ids[idx * self.stride : idx * self.stride + self.context_length]
        target_ids = self.token_ids[idx * self.stride + 1 : idx * self.stride + self.context_length + 1]

        input_ids = torch.tensor(input_ids, dtype = torch.long)
        target_ids = torch.tensor(target_ids, dtype = torch.long)

        return input_ids, target_ids
        raise NotImplementedError("GPTDataset.__getitem__을 구현하세요.")


def create_dataloader(
    token_ids: list[int],
    context_length: int,
    batch_size: int = 8,
    stride: int | None = None,
    drop_last: bool = False,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """TODO: GPTDataset을 만들고 torch.utils.data.DataLoader로 감싸 반환합니다."""
    dataset = GPTDataset(token_ids, context_length, stride)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, drop_last = drop_last, num_workers = num_workers)
    return dataloader
    raise NotImplementedError("create_dataloader를 구현하세요.")
