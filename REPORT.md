# mini GPT 구현 과제 보고서

## 0. 반·팀원

| 항목 | 내용 |
| --- | --- |
| 반 | (예: AI 1반) |
| 팀명 | (예: 3팀) |
| 팀원 | (예: 홍길동, 김철수) |

---

## 1. 구현 현황

| 단계 | 구현 내용 | 구현 파일 | 담당자 |
| --- | --- | --- | --- |
| 1 | UTF-8 byte-level BPE tokenizer | `src/bpe.py` |  |
| 2 | GPTDataset, create_dataloader, InputEmbedding | `src/dataset.py`, `src/embeddings.py` |  |
| 3 | MultiHeadAttention, causal mask | `src/attention.py` |  |
| 4 | LayerNorm, GELU, FeedForward, TransformerBlock, GPTModel, generate_text_simple | `src/model.py` |  |
| 5 | loss 계산, checkpoint, generate, train_model | `src/train.py` |  |
| 6 | NSMC 감성 분류 Dataset과 classifier | `src/finetune.py` |  |

---

## 2. 테스트 통과 현황

| 실행 명령 | 결과 | 비고 |
| --- | --- | --- |
| `pytest tests/test_bpe.py -v` | 통과 / 실패 / 미실행 |  |
| `pytest tests/test_dataset.py -v` | 통과 / 실패 / 미실행 |  |
| `pytest tests/test_attention.py -v` | 통과 / 실패 / 미실행 |  |
| `pytest tests/test_model.py -v` | 통과 / 실패 / 미실행 |  |
| `pytest tests/test_train.py -v` | 통과 / 실패 / 미실행 |  |
| `pytest tests/test_finetune.py -v` | 통과 / 실패 / 미실행 |  |
| `pytest tests/ -v` | 통과 / 실패 / 미실행 |  |

실패한 테스트가 있다면 에러 요약을 적습니다.

| 실패한 테스트 | 에러 요약 | 해결 시도 |
| --- | --- | --- |
| (예: `test_train.py::TestGenerate::test_generate_shape`) |  |  |

---

## 3. 데이터

| 항목 | 내용 |
| --- | --- |
| 원본 데이터 | NSMC |
| 원본 경로 | `data/ratings_train.txt`, `data/ratings_test.txt` |
| 사전 학습 데이터 | `data/nsmc_lm_train.txt`, `data/nsmc_lm_val.txt` |
| 미세 조정 데이터 | `data/nsmc_sentiment_train.jsonl`, `data/nsmc_sentiment_val.jsonl`, `data/nsmc_sentiment_test.jsonl` |
| 전처리 방식 | 빈 리뷰 제거, 공백 정리, train/validation 분리 |
| 사용한 데이터 크기 | Smoke / Light / Basic 중 선택 |

---

## 4. BPE

| 항목 | 내용 |
| --- | --- |
| 구현 파일 | `src/bpe.py` |
| BPE 방식 | UTF-8 byte-level BPE |
| 특수 토큰 ID | `<pad>=0`, `<unk>=1`, `<bos>=2`, `<eos>=3` |
| byte token ID 범위 | 4~259 |
| vocab_size | (예: 3000) |
| 학습 corpus 크기 | (예: `corpus[:1_500_000]`) |
| 어휘 학습 시간 | (예: Colab CPU Basic 설정 35분) |
| vocabulary 저장 경로 | (예: `data/nsmc_bpe_vocab_3000.json`) |
| 인코딩/디코딩 복원 예시 | (예: `decode(encode("이 영화는 좋았다")) == 원문`) |

---

## 5. 모델 구조

| 항목 | 내용 |
| --- | --- |
| 구현 파일 | `src/model.py` |
| 전체 구조 | InputEmbedding -> N x TransformerBlock -> LayerNorm -> LM head |
| vocab_size | (예: 3000) |
| context_length | (예: 128) |
| emb_dim | (예: 192) |
| n_heads | (예: 4) |
| n_layers | (예: 4) |
| drop_rate | (예: 0.1) |
| qkv_bias | True / False |
| 총 파라미터 수 | (계산식 포함) |

---

## 6. 사전 학습

### 6.1 하이퍼파라미터

| 구분 | 항목 | 값 |
| --- | --- | --- |
| 모델 | vocab_size |  |
| 모델 | context_length |  |
| 모델 | emb_dim |  |
| 모델 | n_heads |  |
| 모델 | n_layers |  |
| 학습 | batch_size |  |
| 학습 | num_epochs |  |
| 학습 | eval_freq, eval_iter |  |
| 최적화 | lr, weight_decay |  |

### 6.2 결과

| 항목 | 내용 |
| --- | --- |
| train loss | epoch별 표 또는 요약 |
| validation loss | epoch별 표 또는 요약 |
| 손실 그래프 | 그래프 또는 파일 경로 |
| 생성 샘플 | 같은 시작 문맥으로 epoch별 비교 |
| checkpoint 경로 | (예: `checkpoints/ckpt_epoch_5.pt`) |

---

## 7. 미세 조정

| 항목 | 내용 |
| --- | --- |
| 구현 파일 | `src/finetune.py` |
| 과제 | NSMC 리뷰 긍정/부정 분류 |
| 데이터 포맷 | JSONL, `text`, `label` |
| max_length | (예: 128) |
| batch_size | (예: 16) |
| backbone learning rate |  |
| classifier learning rate |  |
| validation loss / accuracy |  |
| test loss / accuracy |  |
| 오류 예시 | 틀린 리뷰 예시와 추정 원인 |

---

## 8. 실험 환경

| 항목 | 내용 |
| --- | --- |
| Python | (예: Python 3.11) |
| PyTorch | (예: PyTorch 2.x) |
| 실행 환경 | Colab GPU / Colab CPU / 로컬 |
| GPU/CPU 정보 |  |
| 총 학습 소요 시간 |  |

---

## 9. 고찰

### GPT 모델 성능 개선 요약

우리 팀은 GPT 성능 개선을 모델 크기만 키우는 방식으로 접근하지 않았다. 먼저 loss가 잘 내려가지 않는 현상을 관찰하고, 그 원인을 tokenizer 난도, learning rate, context length, model capacity, regularization, 문장 경계 정보로 나누어 가설을 세웠다. 이후 한 번에 하나의 변수를 바꾸며 train loss, validation loss, train-val gap, 생성 샘플을 비교했다.

### 가설 -> 결론 루프

| 단계 | 가설 | 결론 |
| --- | --- | --- |
| 초기 baseline | `vocab_size=10000`은 작은 GPT와 NSMC corpus에 비해 너무 커서 token 예측이 어렵다. | 2M train chars, 2000 steps에서도 val loss가 `8.116` 수준에 머물러 tokenizer 난도 축소가 필요했다. |
| vocab size 축소 | vocab을 `3000`으로 줄이면 token sparsity가 줄고 학습이 쉬워진다. | vocab 3000 실험군에서 loss가 5점대까지 내려가 이후 실험의 기준 tokenizer가 됐다. |
| learning rate sweep | learning rate가 너무 낮거나 높으면 수렴이 나빠진다. | `lr=0.002`에서 final val loss `5.249`로 가장 안정적인 결과를 얻었다. |
| context length 비교 | context를 길게 하면 더 많은 문맥을 보고 예측할 수 있다. | context 128은 update step 수가 줄어 val loss `5.875`로 나빴고, 최종적으로 `context_length=16`을 절충값으로 선택했다. |
| model capacity 확장 | `emb=128`, `layers=2`는 표현력이 부족하다. | `emb=256`, `heads=8`, `layers=4`에서 val loss `5.370`으로 개선되어 capacity 부족 가설이 지지됐다. |
| dropout / weight decay 조정 | capacity 증가 후 regularization을 조정하면 validation loss가 낮아질 수 있다. | 현재 조건에서는 dropout이 학습을 방해했고, `drop_rate=0.0`, `weight_decay=0.1`이 더 나았다. |
| BOS/EOS 경계 token | 리뷰 사이 경계를 `<bos>`, `<eos>`로 알려주면 next-token target이 쉬워진다. | 실제 run에서 final validation loss가 `4.745`까지 내려가 가장 큰 후반부 개선이 됐다. |

### 대표 그래프

![BOS EOS actual validation loss 4.745](assets/gpt-training-analysis/bos_eos_actual_val4745.png)

### 최종 해석

- 한국어 byte-level BPE는 글자 단위가 아니라 UTF-8 byte 단위로 처리해야 decode 안정성을 유지할 수 있었다.
- 초기 loss 정체의 핵심 원인은 모델 구조 하나가 아니라 tokenizer 난도, learning rate, context length, capacity, regularization, 문장 경계 정보가 함께 얽힌 문제였다.
- 가장 큰 후반부 개선은 BOS/EOS 문장 경계 token이었다. 리뷰와 리뷰 사이의 불연속 전이를 모델이 억지로 학습하지 않아도 되면서 validation loss가 `4.745`까지 내려갔다.
- 다음 개선 후보는 warmup/cosine decay의 체계적 비교, BOS/EOS token cache key 분리, 생성 품질 정량 평가, checkpoint별 validation sample 분석이다.
