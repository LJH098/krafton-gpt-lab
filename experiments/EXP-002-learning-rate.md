# EXP-002: learning_rate 비교

## 1. 배경

현재 loss가 느리게 내려가거나 특정 구간 이후 validation loss 개선이 둔화될 수 있다. 학습률이 너무 낮으면 학습이 느리고, 너무 높으면 loss가 출렁이거나 validation loss가 나빠질 수 있다.

---

## 2. 가설

```text
현재 learning_rate=3e-4가 최적이 아닐 수 있다.
1e-4는 더 안정적이지만 느릴 수 있고, 5e-4는 더 빠르게 loss를 낮추지만 불안정할 수 있다.

따라서 같은 데이터/model/tokenizer 조건에서 learning_rate만 바꿔 비교하면
loss 하강 속도와 안정성의 균형점을 찾을 수 있다.
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
| `learning_rate` / `PRETRAIN_LR` | `3e-4` | `1e-4`, `5e-4` |
| `model_size` | `emb_dim=128`, `layers=2`, `heads=4` | 동일 |

### 고정하는 것

| 항목 | 값 |
| --- | --- |
| tokenizer | `data/bpe_tokenizer_vocab3000.json` |
| batch_size | `16` |
| context_length | `64` |
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
| emb_dim | `128` |
| n_heads | `4` |
| n_layers | `2` |
| drop_rate | `0.1` |
| context_length | `64` |
| batch_size | `16` |
| learning_rate | `1e-4` / `3e-4` / `5e-4` |

---

## 7. 성공 기준

- baseline보다 같은 epoch에서 `val loss`가 낮으면 성공
- loss 곡선이 크게 튀지 않아야 한다
- 최종 `val loss`가 baseline보다 `0.1` 이상 낮으면 유지 후보
- train loss만 낮고 val loss가 높으면 실패

---

## 8. 결과

| learning_rate | step | train loss | val loss | 비고 |
| --- | --- | --- | --- | --- |
| `1e-4` | 0 |  |  | 시작 |
| `1e-4` | final |  |  | 종료 |
| `3e-4` | 0 |  |  | baseline |
| `3e-4` | final |  |  | baseline |
| `5e-4` | 0 |  |  | 시작 |
| `5e-4` | final |  |  | 종료 |

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

- EXP-003: 가장 좋은 `PRETRAIN_LR`로 `context_length=128` 비교
- EXP-005: 가장 좋은 `PRETRAIN_LR`로 regularization 비교

---

## 핵심 체크리스트

- [ ] 가설
- [ ] 변경 변수
- [ ] 고정 변수
- [ ] 성공 기준
- [ ] 결론과 다음 실험
