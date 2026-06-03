# EXP-001: vocab_size 10000 vs 3000

## 1. 배경

현재 pretraining에서 생성 문장이 조사/빈출 단어 위주로 반복되거나, validation loss가 느리게 내려가는 문제가 있다.

이전에 `vocab_size=10000` tokenizer를 사용했을 때 현재 corpus와 모델 크기에 비해 token 예측 난도가 높았을 가능성이 있다. 현재 repo에는 `data/bpe_tokenizer_vocab3000.json`이 있으며, 노트북은 이 파일을 기준으로 `vocab_size=3000`을 사용하도록 되어 있다.

---

## 2. 가설

```text
vocab_size=10000은 현재 corpus와 모델 크기에 비해 너무 커서 token 예측 난도가 높고,
token 희소성이 커져 loss가 느리게 내려간다.

따라서 같은 corpus/model 조건에서 vocab_size를 3000으로 줄이면
val loss가 더 빠르게 내려가고 생성 품질도 개선될 것이다.
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
| `vocab_size` | `10000` | `3000` |
| `learning_rate` / `PRETRAIN_LR` | `3e-4` | `3e-4` |
| `model_size` | `emb_dim=128`, `layers=2`, `heads=4` | 동일 |

### 고정하는 것

| 항목 | 값 |
| --- | --- |
| tokenizer | BPE tokenizer, merge 개수만 다름 |
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
| learning_rate | `3e-4` |

---

## 7. 성공 기준

- `vocab_size=3000`이 baseline보다 `val loss`를 더 빠르게 낮추면 성공
- 최종 `val loss`가 baseline보다 `0.2` 이상 낮으면 성공
- `train loss`와 `val loss` 차이가 `0.3` 이하이면 유지 가능
- 생성 샘플에서 조사/어미 반복이 줄면 긍정적 신호

---

## 8. 결과

| step | train loss | val loss | 비고 |
| --- | --- | --- | --- |
| 0 |  |  | 시작 |
| 100 |  |  |  |
| 500 |  |  |  |
| 1000 |  |  |  |
| final |  |  | 종료 |

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

- EXP-002: `vocab_size=3000` 조건 유지 후 `learning_rate` 비교
- EXP-003: `vocab_size=3000` 조건 유지 후 `context_length=128`
- EXP-004: `vocab_size=3000` 조건 유지 후 model capacity 증가

---

## 핵심 체크리스트

- [ ] 가설
- [ ] 변경 변수
- [ ] 고정 변수
- [ ] 성공 기준
- [ ] 결론과 다음 실험
