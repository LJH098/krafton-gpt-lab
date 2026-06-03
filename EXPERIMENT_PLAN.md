# 실험 계획 인덱스

train/validation loss를 낮추기 위한 실험 기록지는 `experiments/` 폴더에 실험별 Markdown 파일로 분리한다.

## 실험 순서

| 실험 | 파일 | 핵심 변수 |
| --- | --- | --- |
| EXP-001 | `experiments/EXP-001-vocab-size-3000.md` | `vocab_size` |
| EXP-002 | `experiments/EXP-002-learning-rate.md` | `PRETRAIN_LR` |
| EXP-003 | `experiments/EXP-003-context-length-128.md` | `CONTEXT_LENGTH` |
| EXP-004 | `experiments/EXP-004-model-capacity.md` | `PRETRAIN_EMB_DIM`, `PRETRAIN_N_LAYERS` |
| EXP-005 | `experiments/EXP-005-regularization.md` | `PRETRAIN_DROP_RATE`, `PRETRAIN_WEIGHT_DECAY` |
| EXP-006 | `experiments/EXP-006-finetune-settings.md` | fine-tuning settings |

## 공통 원칙

- 한 실험에서는 가능한 한 하나의 가설만 검증한다.
- 변경 변수와 고정 변수를 분리해서 기록한다.
- 성공 기준은 실행 전에 정한다.
- 결과에는 train loss, validation loss, 생성 샘플, 해석, 다음 실험을 반드시 남긴다.
- Colab에서 노트북 값을 바꾼 뒤 실험 파일의 `결과`와 `결정` 섹션을 채운다.

## 핵심 체크리스트

- [ ] 가설
- [ ] 변경 변수
- [ ] 고정 변수
- [ ] 성공 기준
- [ ] 결과
- [ ] 결론과 다음 실험
