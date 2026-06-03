# EXP-008: learning rate warmup

## 1. 배경

초기 step에서 큰 learning rate를 바로 쓰면 loss가 출렁이거나 불안정해질 수 있다. Linear warmup으로 작은 lr에서 시작해 목표 lr까지 올리면 초반 학습을 안정화할 수 있다.

---

## 2. 가설

```text
전체 pretraining step의 5% 동안 learning rate를 선형 warmup하면
초기 loss 변동이 줄고 validation loss가 baseline보다 안정적으로 낮아질 수 있다.
```

---

## 3. 실험 설계

### 변경하는 것

| 항목 | Baseline | Experiment |
| --- | --- | --- |
| `PRETRAIN_WARMUP_RATIO` | `0.0` | `0.05` |
| `PRETRAIN_WARMUP_STEPS` | `0` | `len(train_loader) * NUM_EPOCHS * 0.05` |
| lr schedule | constant lr | linear warmup 후 constant lr |

### 고정하는 것

| 항목 | 값 |
| --- | --- |
| tokenizer / BOS-EOS 정책 | 현재 선택한 설정과 동일 |
| `TRAIN_CHARS`, `VAL_CHARS` | baseline과 동일 |
| `PRETRAIN_LR` | baseline과 동일 |
| model size | baseline과 동일 |
| batch / context / stride | baseline과 동일 |
| optimizer / weight_decay | baseline과 동일 |

---

## 4. 성공 기준

- 초기 eval 구간에서 train/validation loss 변동이 줄면 긍정 신호
- final validation loss가 baseline보다 낮으면 Keep 후보
- warmup 때문에 학습이 너무 느려지면 ratio를 `0.02` 또는 `0.03`으로 재실험

---

## 5. 결과 기록

| 항목 | Baseline | Warmup |
| --- | --- | --- |
| total steps |  |  |
| warmup steps |  |  |
| final train loss |  |  |
| final val loss |  |  |
| loss graph |  |  |
| 생성 샘플 |  |  |

---

## 6. 결정

- [ ] Keep
- [ ] Reject
- [ ] Retry

### 결정 이유

```text

```
