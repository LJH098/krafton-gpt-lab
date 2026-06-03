# EXP-004: 모델 크기 확장

## 1. 배경

현재 모델이 `emb_dim=128`, `n_layers=2`로 작기 때문에 train loss 자체가 충분히 내려가지 않을 수 있다. 모델 용량이 부족하면 train loss와 validation loss가 둘 다 높은 상태로 머무를 수 있다.

---

## 2. 가설

```text
현재 모델 capacity가 corpus/tokenizer에 비해 작아 패턴을 충분히 학습하지 못하고 있다.

따라서 emb_dim과 layer 수를 늘리면 train loss가 더 낮아지고,
과적합이 심하지 않다면 validation loss도 함께 낮아질 것이다.
```

---

## 3. 실험 설계

### 변경하는 것

| 항목 | Baseline | Experiment A | Experiment B |
| --- | --- | --- | --- |
| `MAX_TRAIN_CHARS` / `TRAIN_CHARS` | `1_500_000` | `1_500_000` | `1_500_000` |
| `MAX_VAL_CHARS` / `VAL_CHARS` | `100_000` | `100_000` | `100_000` |
| `num_epochs` / `NUM_EPOCHS` | `8` | `8` | `8` |
| `max_steps` | 전체 epoch | 전체 epoch | 전체 epoch |
| `vocab_size` | `3000` | `3000` | `3000` |
| `learning_rate` | EXP-002 best | EXP-002 best | EXP-002 best |
| `model_size` | `128/4/2` | `192/4/4` | `256/8/4` |

### 고정하는 것

| 항목 | 값 |
| --- | --- |
| tokenizer | `data/bpe_tokenizer_vocab3000.json` |
| batch_size | `16`, OOM이면 `8` |
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
| emb_dim | `192` or `256` |
| n_heads | `4` or `8` |
| n_layers | `4` |
| drop_rate | `0.1` |
| context_length | EXP-003 best |
| batch_size | `16` or `8` |
| learning_rate | EXP-002 best |

---

## 7. 성공 기준

- train loss가 baseline보다 확실히 낮아지면 capacity 증가는 효과 있음
- validation loss도 낮아지면 Keep
- train loss만 낮아지고 val loss가 높아지면 과적합으로 판단
- OOM이 나면 batch size를 줄여 Retry

---

## 8. 결과

| model | step | train loss | val loss | 비고 |
| --- | --- | --- | --- | --- |
| `128/4/2` | final |  |  | baseline |
| `192/4/4` | final |  |  |  |
| `256/8/4` | final |  |  |  |

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

- EXP-005: 선택한 model size에서 regularization 조정
- EXP-006: 선택한 backbone으로 fine-tuning 성능 확인

---

## 핵심 체크리스트

- [ ] 가설
- [ ] 변경 변수
- [ ] 고정 변수
- [ ] 성공 기준
- [ ] 결론과 다음 실험
