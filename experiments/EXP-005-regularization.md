# EXP-005: dropout / weight_decay 조정

## 1. 배경

모델 크기나 학습 시간을 늘리면 train loss는 낮아질 수 있지만 validation loss가 따라오지 않으면 과적합이다. 반대로 regularization이 너무 강하면 train loss도 충분히 내려가지 않는다.

---

## 2. 가설

```text
현재 dropout 또는 weight_decay가 최적이 아닐 수 있다.

dropout과 weight_decay를 조정하면 train loss와 validation loss의 차이를 줄이고,
최종 validation loss를 낮출 수 있다.
```

---

## 3. 실험 설계

### 변경하는 것

| 항목 | Baseline | Experiment |
| --- | --- | --- |
| `MAX_TRAIN_CHARS` / `TRAIN_CHARS` | `1_500_000` | `1_500_000` |
| `MAX_VAL_CHARS` / `VAL_CHARS` | `100_000` | `100_000` |
| `num_epochs` / `NUM_EPOCHS` | `8` | `8` |
| `max_steps` | 전체 epoch | 전체 epoch |
| `vocab_size` | `3000` | `3000` |
| `learning_rate` | EXP-002 best | EXP-002 best |
| `model_size` | EXP-004 best | EXP-004 best |
| `drop_rate` | `0.1` | `0.0`, `0.2` |
| `weight_decay` | `0.1` | `0.01`, `0.05` |

### 고정하는 것

| 항목 | 값 |
| --- | --- |
| tokenizer | `data/bpe_tokenizer_vocab3000.json` |
| batch_size | EXP-004 best |
| context_length | EXP-003 best |
| optimizer | `AdamW` |
| eval_freq | `max(1, len(train_loader) // 2)` |
| eval_iter | `min(20, len(val_loader))` |
| seed | 기록 필요 |

---

## 4. 실행 환경

| 항목 | 값 |
| --- | --- |
| 실행 위치 | Colab |
| GPU | Tesla T4 / 기타:  |
| Python |  |
| PyTorch |  |
| 실행 시작 시간 |  |
| 실행 소요 시간 |  |

---

## 5. 데이터 조건

| 항목 | 값 |
| --- | --- |
| train chars | `1_500_000` |
| val chars | `100_000` |
| train tokens |  |
| val tokens |  |
| steps per epoch |  |
| actual steps |  |

---

## 6. 모델 설정

| 항목 | 값 |
| --- | --- |
| vocab_size | `3000` |
| emb_dim | EXP-004 best |
| n_heads | EXP-004 best |
| n_layers | EXP-004 best |
| drop_rate | `0.0` / `0.1` / `0.2` |
| context_length | EXP-003 best |
| batch_size | EXP-004 best |
| learning_rate | EXP-002 best |

---

## 7. 성공 기준

- validation loss가 baseline보다 낮으면 성공
- train loss와 val loss 차이가 `0.3` 이하이면 유지 가능
- train loss가 너무 높아지면 regularization이 과하다고 판단
- 생성 샘플 반복이 줄면 긍정적 신호

---

## 8. 결과

| drop_rate | weight_decay | train loss | val loss | 비고 |
| --- | --- | --- | --- | --- |
| `0.1` | `0.1` |  |  | baseline |
| `0.1` | `0.01` |  |  |  |
| `0.1` | `0.05` |  |  |  |
| `0.0` | best |  |  |  |
| `0.2` | best |  |  |  |

### 최종 결과

| 항목 | 값 |
| --- | --- |
| 최종 train loss |  |
| 최종 val loss |  |
| perplexity |  |
| token accuracy |  |
| loss graph |  |
| checkpoint |  |

---

## 9. 생성 샘플

### Prompt

```text
이 영화는
```

### Output

```text

```

### 관찰

- 반복이 있는가?
- 문장 경계가 자연스러운가?
- 조사/어미가 과하게 반복되는가?
- UTF-8 decode 문제가 있는가?

---

## 10. 해석

```text

```

---

## 11. 결정

- [ ] Keep
- [ ] Reject
- [ ] Retry
- [ ] Follow-up

### 결정 이유

```text

```

---

## 12. 다음 실험

- EXP-006: 선택한 pretraining checkpoint/backbone으로 sentiment fine-tuning

---

## 핵심 체크리스트

- [ ] 가설
- [ ] 변경 변수
- [ ] 고정 변수
- [ ] 성공 기준
- [ ] 결론과 다음 실험
