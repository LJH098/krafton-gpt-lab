# -*- coding: utf-8 -*-
"""
UTF-8 byte-level BPE 토크나이저 과제 템플릿.

외부 tokenizer 라이브러리 없이 BPE(Byte Pair Encoding)를 직접 구현합니다.
한국어 NSMC 리뷰를 다루므로 문자열을 글자/공백 단위로 먼저 자르지 말고,
항상 `text.encode("utf-8")`로 byte ID 시퀀스를 만든 뒤 merge를 적용하세요.
"""

from pathlib import Path
import json


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
SPECIAL_IDS = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
BYTE_OFFSET = len(SPECIAL_TOKENS)
NUM_BYTES = 256


class BPETokenizer:
    """
    UTF-8 byte-level BPE 토크나이저.

    권장 ID 배치:
    - 0~3: <pad>, <unk>, <bos>, <eos>
    - 4~259: 원본 byte 0~255
    - 260 이상: BPE merge로 생성한 토큰
    """

    def __init__(self, vocab_size: int = 3000):
        self.vocab_size = vocab_size
        self.id_to_token = {}
        self.token_to_id = {}
        self.merges = []

    def _init_special_tokens(self):
        """
        TODO:
        1. 특수 토큰 4개를 고정 ID 0~3에 등록합니다.
        2. byte 0~255를 ID 4~259에 bytes([byte_value]) 형태로 등록합니다.
        """
        for id in range(0, 3 + 1):
            self.id_to_token[id] = SPECIAL_TOKENS[id]
            self.token_to_id[SPECIAL_TOKENS[id]] = id
        for i in range(NUM_BYTES):
            token_id = i + BYTE_OFFSET
            token = bytes([i])

            self.id_to_token[token_id] = token
            self.token_to_id[token] = token_id


    def get_pad_id(self):
        """padding 토큰 ID."""
        return SPECIAL_IDS[PAD_TOKEN]

    def get_unk_id(self):
        """unknown 토큰 ID."""
        return SPECIAL_IDS[UNK_TOKEN]

    def get_bos_id(self):
        """문장 시작 토큰 ID."""
        return SPECIAL_IDS[BOS_TOKEN]

    def get_eos_id(self):
        """문장 끝 토큰 ID."""
        return SPECIAL_IDS[EOS_TOKEN]

    def train(self, corpus: str):
        """
        TODO: 코퍼스에서 BPE merge rule과 vocabulary를 학습합니다.

        구현 힌트:
        - `corpus.encode("utf-8")`로 byte ID 시퀀스를 만듭니다.
        - 가장 자주 등장하는 이웃 token pair를 찾습니다.
        - 새 token ID를 만들고, 시퀀스의 해당 pair를 새 ID로 치환합니다.
        - `self.merges`, `self.id_to_token`, `self.token_to_id`를 갱신합니다.
        """
        self._init_special_tokens()
        bytes_list = list(corpus.encode("utf-8"))
        bytes_list = [b + BYTE_OFFSET for b in bytes_list]
        while len(self.id_to_token) < self.vocab_size and len(bytes_list) > 1:
            freq = {}
            for i in range(len(bytes_list) - 1):
                pair = (bytes_list[i], bytes_list[i + 1])
                freq[pair] = freq.get(pair, 0) + 1
            
            most_freq_pair = max(freq, key=lambda pair: freq[pair])

            id = len(self.id_to_token)

            i = 0
            while i < len(bytes_list):
                if i + 1 < len(bytes_list) and (bytes_list[i], bytes_list[i + 1]) == most_freq_pair:
                    bytes_list[i] = id
                    del bytes_list[i + 1]
                i += 1
            # merges, id_to_token, token_to_id에 넣기
            self.merges.append(most_freq_pair)
            self.id_to_token[id] = most_freq_pair
            self.token_to_id[most_freq_pair] = id

    def save(self, path: str | Path):
            """
            TODO: vocabulary와 merge rule을 JSON 파일로 저장합니다.

            bytes와 tuple은 JSON에 바로 저장할 수 없으므로 type 정보를 함께 저장하세요.
            """
            data = {}
            merges_data = []

            for tup in self.merges:
                merges_data.append({
                    "type" : "tuple",
                    "value" : list(tup)
                })


            data["merges"] = merges_data
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)


    def load(self, path: str | Path):
        """
        TODO: save()로 저장한 JSON 파일을 읽어 vocabulary와 merge rule을 복원합니다.
        """
        # raise NotImplementedError("BPETokenizer.load를 구현하세요.")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._init_special_tokens()
        self.merges = []

        data_list = data["merges"]

        new_id = len(self.id_to_token)
        for dic in data_list:
            token = tuple(dic["value"])
            self.id_to_token[new_id] = token
            self.token_to_id[token] = new_id
            self.merges.append(token)
            new_id += 1

    def encode(self, text: str, add_bos_eos: bool = False) -> list[int]:
        """
        TODO: 문자열을 token ID 리스트로 변환합니다.

        구현 힌트:
        - 먼저 UTF-8 byte ID 리스트를 만듭니다.
        - train/load에서 얻은 merge rule을 학습 순서대로 적용합니다.
        - add_bos_eos=True이면 앞뒤에 bos/eos ID를 붙입니다.
        """
        text_encoded = text.encode("utf-8")
        token_ids = []
        token_ids.append(SPECIAL_IDS[BOS_TOKEN])
        for te in text_encoded:
            token_ids.append(self.token_to_id[bytes([te])])
        
        for merge_rule in self.merges:
            merge_list = []
            # for i in range(len(token_ids)):
            i = 1
            while i < len(token_ids):
                if i + 1 < len(token_ids) and (token_ids[i], token_ids[i + 1]) == merge_rule:
                    merge_list.append(self.token_to_id[merge_rule])
                    i += 2
                else:
                    merge_list.append(token_ids[i])
                    i += 1
            token_ids = merge_list
        token_ids.append(SPECIAL_IDS[EOS_TOKEN])
        return token_ids





    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """
        TODO: token ID 리스트를 문자열로 복원합니다.

        주의:
        - merge token은 원본 byte token까지 재귀적으로 펼칩니다.
        - byte를 하나씩 decode하지 말고, 마지막에 `bytes(...).decode("utf-8")`를 한 번만 호출합니다.
        """
        # ids = []
        IDS = []
        def divide(id):
            if id < 260:
                IDS.append(id)
            else :
                p1, p2 = self.id_to_token[id]
                divide(p1)
                divide(p2)
        for id in ids:
            divide(id)
        
        tokens = []
        for id in IDS:
            if skip_special and id <= 3:
                continue
            tokens.append(self.id_to_token[id])
        
        return b"".join(tokens).decode("utf-8")
