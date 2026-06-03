# EXP-006: sentiment fine-tuning 설정 비교

## 1. 배경

Pretraining loss가 낮아져도 sentiment classification 성능이 자동으로 좋아지는 것은 아니다. Classification head와 backbone fine-tuning 방식, max_length, learning rate가 validation/test accuracy에 큰 영향을 줄 수 있다.

---

## 2. 가설

```text
현재 fine-tuning 설정이 backbone을 충분히 활용하지 못하거나 과적합을 만들 수 있다.

FREEZE_BACKBONE, FINETUNE_LR, SENTIMENT_MAX_LENGTH_CAP을 비교하면
validation/test accuracy가 더 좋은 fine-tuning 조건을 찾을 수 있다.
```

---

## 3. 실험 설계

### 변경하는 것

| 항목 | Baseline | Experiment |
| --- | --- | --- |
| `SENTIMENT_TRAIN_LIMIT` | `20_000` | `80_000` |
| `SENTIMENT_VAL_LIMIT` | `5_000` | `10_000` |
| `num_epochs` / `FINETUNE_NUM_EPOCHS` | `3` | `3` or `5` |
| `max_steps` | 전체 epoch | 전체 epoch |
| `vocab_size` | `3000` | `3000` |
| `learning_rate` / `FINETUNE_LR` | `1e-4` | `5e-5`, `3e-4` |
| `model_size` | selected pretraining backbone | 동일 |
| `FREEZE_BACKBONE` | `False` | `True` |

### 고정하는 것

| 항목 | 값 |
| --- | --- |
| tokenizer | `data/bpe_tokenizer_vocab3000.json` |
| batch_size | `16` |
| context_length | selected pretraining setting |
| optimizer | `AdamW` |
| eval_freq | epoch마다 validation |
| eval_iter | 전체 validation loader |
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
| train examples |  |
| val examples |  |
| test examples |  |
| max_length |  |
| steps per epoch |  |
| actual steps |  |

---

## 6. 모델 설정

| 항목 | 값 |
| --- | --- |
| vocab_size | `3000` |
| emb_dim | selected pretraining setting |
| n_heads | selected pretraining setting |
| n_layers | selected pretraining setting |
| drop_rate | selected pretraining setting |
| classifier_drop_rate | `0.1` |
| context_length | selected pretraining setting |
| batch_size | `16` |
| learning_rate | `5e-5` / `1e-4` / `3e-4` |

---

## 7. 성공 기준

- validation accuracy가 baseline보다 높으면 성공
- test accuracy가 baseline보다 높으면 최종 Keep 후보
- validation loss가 오르는데 train accuracy만 높으면 과적합
- `FREEZE_BACKBONE=True`가 성능이 비슷하면 빠른 설정으로 유지 가능

---

## 8. 결과

| setting | train loss | train acc | val loss | val acc | test acc | 비고 |
| --- | --- | --- | --- | --- | --- | --- |
| baseline |  |  |  |  |  |  |
| freeze backbone |  |  |  |  |  |  |
| lr `5e-5` |  |  |  |  |  |  |
| lr `3e-4` |  |  |  |  |  |  |
| train limit `80_000` |  |  |  |  |  |  |

### 최종 결과

| 항목 | 값 |
| --- | --- |
| 최종 train loss |  |
| 최종 val loss |  |
| validation accuracy |  |
| test accuracy |  |
| loss graph |  |
| checkpoint |  |

---

## 9. 예측 샘플

### Input

```text
이 영화는 배우들의 연기가 좋고 끝까지 몰입됐다.
```

### Output

```text

```

### 관찰

- 긍정/부정 예측이 맞는가?
- 확신도가 과하게 높지 않은가?
- 짧은 리뷰와 긴 리뷰에서 차이가 있는가?
- 부정어가 포함된 문장을 잘 처리하는가?

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

- EXP-007: confusion examples 수집
- EXP-008: backbone freeze 후 classifier만 더 오래 학습
- EXP-009: max_length 128 vs 256 비교

---

## 핵심 체크리스트

- [ ] 가설
- [ ] 변경 변수
- [ ] 고정 변수
- [ ] 성공 기준
- [ ] 결론과 다음 실험
