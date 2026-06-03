# EXP-007: BOS/EOS line boundary pretraining

## 1. 배경

NSMC LM corpus는 리뷰 하나가 한 줄에 저장되어 있다. 기존 pretraining tokenization은 corpus 전체를 한 번에 encode하므로 리뷰 시작과 끝을 모델이 명시적으로 볼 수 없다.

---

## 2. 가설

```text
각 리뷰 line 앞뒤에 <bos>/<eos>를 붙이면 모델이 리뷰 경계를 학습하고,
validation loss와 generation 품질이 baseline보다 좋아질 수 있다.
```

---

## 3. 실험 설계

### 변경하는 것

| 항목 | Baseline | Experiment |
| --- | --- | --- |
| `ADD_BOS_EOS_PER_LINE` | `False` | `True` |
| `SKIP_EMPTY_LM_LINES` | `True` | `True` |
| LM tokenization | corpus 전체 encode | line별 `add_bos_eos=True` 후 concat |
| token cache key | no boundary policy | `bosline1_skipempty1` 포함 |

### 고정하는 것

| 항목 | 값 |
| --- | --- |
| tokenizer | `data/bpe_tokenizer_vocab3000.json` |
| `TRAIN_CHARS` | baseline과 동일 |
| `VAL_CHARS` | baseline과 동일 |
| `CONTEXT_LENGTH` | baseline과 동일 |
| `BATCH_SIZE` | baseline과 동일 |
| model size | baseline과 동일 |
| optimizer / lr / weight_decay | baseline과 동일 |
| `NUM_EPOCHS` | baseline과 동일 |

---

## 4. 성공 기준

- validation loss가 baseline보다 낮으면 Keep 후보
- train loss만 낮고 validation loss가 나빠지면 Reject 또는 재실험
- 생성 샘플에서 리뷰 경계가 더 자연스러워지면 긍정 신호

---

## 5. 결과 기록

| 항목 | Baseline | BOS/EOS line |
| --- | --- | --- |
| train tokens |  |  |
| val tokens |  |  |
| steps per epoch |  |  |
| final train loss |  |  |
| final val loss |  |  |
| loss graph |  |  |
| token cache path |  |  |

### 생성 샘플

| start context | Baseline | BOS/EOS line |
| --- | --- | --- |
| `이 영화는` |  |  |
| `정말` |  |  |

---

## 6. 결정

- [ ] Keep
- [ ] Reject
- [ ] Retry

### 결정 이유

```text

```
