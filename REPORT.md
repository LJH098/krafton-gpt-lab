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

### 6.1 최종 대표 설정

사전 학습은 NSMC 리뷰를 language modeling corpus로 사용해 Mini GPT가 다음 token을 예측하도록 학습했다. 실험은 `vocab_size`, `learning_rate`, `context_length`, 모델 크기, regularization, BOS/EOS 문장 경계 token을 순서대로 바꾸며 진행했다.

| 구분 | 항목 | 값 |
| --- | --- | --- |
| tokenizer | 방식 | UTF-8 byte-level BPE |
| tokenizer | vocab_size | `3000` |
| tokenizer | special token | `<pad>=0`, `<unk>=1`, `<bos>=2`, `<eos>=3` |
| 데이터 | train chars | `1_500_000` 기준 실험 |
| 데이터 | val chars | `100_000` 기준 실험 |
| 모델 | context_length | `16` 최종 채택 |
| 모델 | emb_dim / heads / layers | `256 / 8 / 4` 최종 채택 |
| 모델 | dropout | `0.0` 최종 채택 |
| 학습 | batch_size | CUDA 기준 `16` |
| 학습 | num_epochs | `8` 기준 |
| 최적화 | optimizer | AdamW |
| 최적화 | learning_rate | `0.002` 최종 채택 |
| 최적화 | weight_decay | `0.1` |

### 6.2 최종 성능 요약

초기 vocab 10000 baseline에서는 validation loss가 `8.1` 근처에서 정체됐다. 이후 vocab 축소, learning rate 조정, context length 재검토, 모델 capacity 확장, dropout 제거를 거치며 validation loss를 5점대까지 낮췄다. 마지막으로 BOS/EOS 문장 경계 token을 넣은 실제 run에서는 final validation loss가 `4.745`까지 내려갔다.

![BOS EOS actual validation loss 4.745](assets/gpt-training-analysis/bos_eos_actual_val4745.png)

| 단계 | 대표 변화 | validation loss |
| --- | --- | ---: |
| 초기 baseline | vocab 10000, 2M train chars, 2000 steps | `8.116` |
| vocab 축소 후 | vocab 3000 실험군 | 5점대 진입 |
| learning rate sweep | `lr=0.002` 채택 | `5.249` |
| model capacity 확장 | `emb=256`, `heads=8`, `layers=4` | `5.370` |
| dropout 제거 | `drop_rate=0.0` 계열 | `5.505` 계열 |
| BOS/EOS 적용 | 리뷰 단위 문장 경계 token 추가 | `4.745` |

## 7. GPT 모델 성능 개선 루프

### 7.1 실험 방식

팀의 개선 과정은 다음 루프로 진행됐다.

```text
1. 현상 관찰
   loss가 너무 높음, validation loss가 정체됨, 생성 문장이 반복됨, train-val gap이 커짐

2. 병목 가설 설정
   vocab이 너무 큰가?
   learning rate가 부적절한가?
   context length가 현재 데이터에 맞지 않는가?
   모델 capacity가 부족한가?
   dropout/weight_decay가 과한가?
   리뷰 단위 문장 경계가 없어 next-token target이 어려운가?

3. 변수 통제 실험
   vocab_size, learning_rate, context_length, emb_dim/layers/heads,
   drop_rate/weight_decay, BOS/EOS 경계 토큰 등을 하나씩 바꿔 비교

4. 동일 지표 평가
   train loss, validation loss, train-val gap, 생성 샘플 확인

5. 결론 반영
   더 좋은 설정은 notebook config와 실험 문서에 반영하고,
   다음 병목을 새 가설로 세움
```

이 방식의 장점은 loss 하나만 본 것이 아니라, `train loss와 validation loss의 차이`, `생성 샘플`, `그래프 메타데이터`, `다음 실험 계획`을 함께 남겼다는 점이다.

### 7.2 초기 baseline: vocab 10000에서 loss 정체

![Vocab 10000 baseline](assets/gpt-training-analysis/vocab10000_2m200k_steps2000.png)

| 항목 | 내용 |
| --- | --- |
| 가설 | 초기 `vocab_size=10000`은 작은 GPT와 NSMC corpus에 비해 너무 커서 token 예측 문제가 어렵다. |
| 실험/관찰 | 2M train chars, 200k val chars, 2000 steps까지 학습했지만 final train loss `8.120`, val loss `8.116` 수준에서 정체됐다. |
| 결론 | 모델 구조를 키우기 전에 tokenizer 난도를 낮춰야 했다. 이 결과가 vocab size 축소 실험의 출발점이 됐다. |

### 7.3 Vocab size 10000 -> 3000

![Vocab 3000 loss curve](assets/gpt-training-analysis/vocab3000_8epoch_curve.png)

| 항목 | 내용 |
| --- | --- |
| 가설 | vocab을 `10000`에서 `3000`으로 줄이면 token sparsity가 줄고 작은 모델이 더 자주 같은 token을 학습할 수 있다. |
| 실험/관찰 | `data/bpe_tokenizer_vocab3000.json`을 사용한 뒤 loss scale이 8대에서 5대까지 내려가는 실험군이 나타났다. |
| 결론 | `vocab_size=3000`은 이후 모든 주요 pretraining 실험의 기준이 됐다. tokenization 난도를 낮춘 것이 첫 번째 큰 개선이었다. |

### 7.4 Learning rate sweep

![Learning rate 0.002](assets/gpt-training-analysis/lr_0_002_steps11790.png)

| 항목 | 내용 |
| --- | --- |
| 가설 | learning rate가 너무 낮으면 느리고, 너무 높으면 수렴이 불안정하다. 현재 조건의 최적 learning rate를 찾아야 한다. |
| 실험/관찰 | `0.0001`, `0.0005`, `0.001`, `0.002`, `0.003`, `0.005`, `0.007`, `0.01`을 비교했다. `0.002`에서 final val loss `5.249`로 가장 좋았다. |
| 결론 | `PRETRAIN_LR=0.002`를 채택했다. `0.003` 이상부터는 개선이 멈추고, `0.005` 이상은 validation loss가 악화됐다. |

### 7.5 Context length 비교

![Context length 128](assets/gpt-training-analysis/context128_steps3144.png)

| 항목 | 내용 |
| --- | --- |
| 가설 | context length를 `64 -> 128`로 늘리면 더 긴 문맥을 보고 예측하므로 validation loss가 낮아질 수 있다. |
| 실험/관찰 | context 128은 final train loss `5.760`, val loss `5.875`로 baseline 64보다 나빴다. context가 길어지며 sample 수와 update step 수가 줄어든 영향이 컸다. |
| 결론 | 현재 NSMC 리뷰와 작은 GPT 조건에서는 긴 context가 항상 이득이 아니었다. loss 기준으로는 짧은 context가 유리했고, 최종적으로 `CONTEXT_LENGTH=16`이 절충값으로 선택됐다. |

### 7.6 Model capacity 확장

![Model capacity 256 8 4](assets/gpt-training-analysis/model_capacity_256_8_4.png)

| 항목 | 내용 |
| --- | --- |
| 가설 | `emb=128`, `layers=2` 모델은 corpus와 tokenizer를 충분히 학습하기에 capacity가 부족하다. 모델을 키우면 train loss와 validation loss가 내려갈 것이다. |
| 실험/관찰 | `emb=192, heads=4, layers=4`와 `emb=256, heads=8, layers=4`를 비교했다. `256/8/4`는 final train loss `4.833`, val loss `5.370`으로 validation 기준 best였다. |
| 결론 | capacity 부족 가설은 지지됐다. 다만 train-val gap이 커졌기 때문에 이후 regularization 실험이 필요해졌다. |

### 7.7 Dropout / weight decay 조정

![Regularization dropout 0 weight decay 0.1](assets/gpt-training-analysis/regularization_dropout00_wd01.png)

| 항목 | 내용 |
| --- | --- |
| 가설 | capacity를 키운 뒤 overfitting이 생길 수 있으므로 dropout과 weight decay를 조정하면 validation loss를 낮출 수 있다. |
| 실험/관찰 | `drop_rate=0.0`에서 train loss와 validation loss가 함께 내려갔다. 반대로 `drop_rate=0.2`는 train loss와 val loss가 모두 높아져 underfitting 신호를 보였다. |
| 결론 | 현재 설정에서는 dropout이 일반화보다 학습 자체를 방해했다. `PRETRAIN_DROP_RATE=0.0`, `PRETRAIN_WEIGHT_DECAY=0.1`을 채택했다. |

### 7.8 BOS/EOS 문장 경계 토큰

![BOS EOS actual validation loss 4.745](assets/gpt-training-analysis/bos_eos_actual_val4745.png)

| 항목 | 내용 |
| --- | --- |
| 가설 | NSMC 리뷰는 독립된 짧은 문장 단위 데이터이므로, 리뷰 사이 경계를 `<bos>`, `<eos>`로 알려주면 next-token target이 쉬워진다. |
| 실험/관찰 | 코드상 `BPETokenizer.encode(..., add_bos_eos=True)`, `get_bos_id`, `get_eos_id`, `generate(..., eos_id=...)`가 확인된다. 첨부된 실제 run 그래프에서는 final validation loss가 `4.745`까지 내려갔다. |
| 결론 | BOS/EOS는 단순한 특수 token이 아니라 데이터 구조를 모델에게 알려준 변경이었다. 앞선 LR, capacity, dropout 조정보다 큰 폭의 후반부 개선을 만들었다. |

### 7.9 발표용 핵심 문장

우리 팀은 GPT 성능 개선을 모델 크기만 키우는 방식으로 접근하지 않았다. 먼저 vocab size를 줄여 token 예측 난도를 낮췄고, learning rate sweep으로 최적화 균형점을 찾았다. 이후 context length, model capacity, dropout/weight decay를 하나씩 비교하며 validation loss를 낮췄다. 마지막으로 BOS/EOS를 넣어 리뷰 단위 문장 경계를 모델에게 알려주자 validation loss가 `4.745`까지 떨어졌고, 이 변경이 가장 극적인 후반부 개선이었다.

---

## 8. 미세 조정

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

## 9. 실험 환경

| 항목 | 내용 |
| --- | --- |
| Python | (예: Python 3.11) |
| PyTorch | (예: PyTorch 2.x) |
| 실행 환경 | Colab GPU / Colab CPU / 로컬 |
| GPU/CPU 정보 |  |
| 총 학습 소요 시간 |  |

---

## 10. 고찰

- 한국어 byte-level BPE는 글자 단위가 아니라 UTF-8 byte 단위로 처리해야 decode 안정성을 유지할 수 있었다.
- 초기 loss 정체의 핵심 원인은 모델 구조 하나가 아니라 tokenizer 난도, learning rate, context length, capacity, regularization, 문장 경계 정보가 함께 얽힌 문제였다.
- 가장 큰 후반부 개선은 BOS/EOS 문장 경계 token이었다. 리뷰와 리뷰 사이의 불연속 전이를 모델이 억지로 학습하지 않아도 되면서 validation loss가 `4.745`까지 내려갔다.
- 다음 개선 후보는 warmup/cosine decay의 체계적 비교, BOS/EOS token cache key 분리, 생성 품질 정량 평가, checkpoint별 validation sample 분석이다.
