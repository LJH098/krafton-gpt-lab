# EXP-003: context_length 64 vs 128

## 1. 배경

현재 `CONTEXT_LENGTH=64`는 짧은 리뷰에는 충분할 수 있지만, 긴 리뷰에서는 앞뒤 문맥을 충분히 보지 못할 수 있다. 문맥 길이를 늘리면 다음 token 예측과 생성 품질이 개선될 가능성이 있다.

---

## 2. 가설

```text
context_length=64는 일부 리뷰의 문맥을 잘라 token 예측에 필요한 정보를 잃게 만든다.

따라서 context_length를 128로 늘리면 더 긴 문맥을 학습할 수 있고,
validation loss와 생성 샘플의 문장 연결성이 개선될 것이다.
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
| `learning_rate` / `PRETRAIN_LR` | EXP-002 best | EXP-002 best |
| `model_size` | `emb_dim=128`, `layers=2`, `heads=4` | 동일 |
| `context_length` | `64` | `128` |

### 고정하는 것

| 항목 | 값 |
| --- | --- |
| tokenizer | `data/bpe_tokenizer_vocab3000.json` |
| batch_size | `16`, OOM이면 `8` |
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
| context_length | `128` |
| batch_size | `16` or `8` |
| learning_rate | EXP-002 best |

---

## 7. 성공 기준

- 최종 `val loss`가 context 64 baseline보다 낮으면 성공
- 같은 epoch에서 loss 하강 속도가 느려져도 최종 val loss가 낮으면 유지 후보
- GPU OOM이 나면 batch size를 줄이고 재시도
- 생성 샘플에서 문장 연결성이 좋아지면 긍정적 신호

---

## 8. 결과

| context_length | step | train loss | val loss | 비고 |
| --- | --- | --- | --- | --- |
| `64` | final |  |  | baseline |
| `128` | 0 |  |  | 시작 |
| `128` | final |  |  | 종료 |

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

- EXP-004: `context_length=128` 유지 후 model capacity 증가
- EXP-005: `context_length=128`에서 dropout/weight decay 조정

---

## 핵심 체크리스트

- [ ] 가설
- [ ] 변경 변수
- [ ] 고정 변수
- [ ] 성공 기준
- [ ] 결론과 다음 실험
