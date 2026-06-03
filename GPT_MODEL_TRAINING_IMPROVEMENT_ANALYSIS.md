# GPT 모델 학습 및 성능 개선 과정 분석

작성일: 2026-06-03  
분석 대상: `jinhyuk` 브랜치 커밋 이력, 레포지토리 코드/문서, 다운로드된 Notion export 문서/CSV/이미지

## 1. 한 줄 결론

우리 팀은 Mini GPT를 단순히 구현하는 데서 끝내지 않고, `tokenizer 크기 -> learning rate -> context length -> model capacity -> regularization -> sentiment fine-tuning` 순서로 병목을 좁혀 가며 실험했다. 핵심 개선 방향은 `vocab_size=3000`, `PRETRAIN_LR=0.002`, 더 큰 backbone인 `emb_dim=256, n_heads=8, n_layers=4`, 그리고 `drop_rate=0.0, weight_decay=0.1` 쪽으로 수렴했다. 다만 `context_length`는 loss 기준으로 짧을수록 유리했지만, 최종 커밋은 문맥 보존과 업데이트 수 사이의 타협으로 보이는 `context_length=16`을 채택했다. fine-tuning은 train 성능은 높아졌지만 validation loss가 상승해, 아직 과적합 제어 루프가 남아 있다.

## 2. 확인한 자료

### 레포지토리

- 브랜치: `jinhyuk`
- 주요 파일:
  - `src/bpe.py`
  - `src/dataset.py`
  - `src/embeddings.py`
  - `src/attention.py`
  - `src/model.py`
  - `src/train.py`
  - `src/finetune.py`
  - `gpt-lab.ipynb`
  - `EXPERIMENT_PLAN.md`
  - `experiments/EXP-001`부터 `EXP-006`
  - `data/bpe_tokenizer_vocab3000.json`

### 외부 문서와 CSV

- `LLM 실험 ... .csv`, `LLM 실험 ... _all.csv`
- `Vocab size 10000 -> 3000`
- `EXP-001 Vocab Size 비교 실험`
- `learning_rate 비교 실험 보고서`
- `context length 64 -> 128`
- `model_capacity`
- `dropout weight_decay 조정`
- `sentiment fine-tuning 설정 비교`
- `EOS BOS`
- `웜업`
- `기본`

### 외부 이미지

사용자가 직접 지정한 PNG와 Notion export 하위 폴더의 PNG까지 모두 확인했다. 세부 값은 이 문서의 부록 표에 정리했다.

## 3. 자료 해석 시 주의할 점

자료를 대조하면 몇 가지 불일치가 있다.

- 일부 초기 이미지에는 파일명이 `bpe_tokenizer_vocab3000.json`으로 보이지만, 그래프 메타데이터의 `vocab_size`는 `10,000`으로 표시된다. 따라서 2026-06-02의 초기 loss 이미지들은 `vocab_size=10000` baseline으로 해석하는 것이 더 타당하다.
- `EXP-001 Vocab Size` 문서는 결과 표가 baseline과 experiment를 완전히 분리해 기록하지 못했다. 정량 결론은 이미지 메타데이터와 이후 `vocab_size=3000` 실험군을 함께 보고 해석해야 한다.
- `context_length` 실험 문서는 제목은 `64 -> 128`이지만, 결과 표에는 `8`, `16`, `32`, `64`, `128`이 함께 들어 있다.
- final notebook commit은 `CONTEXT_LENGTH=16`을 채택했지만, 외부 문서에서 `16` 선택 이유를 명시적으로 결론 내리지는 않았다. 이 문서에서는 loss와 문맥 길이의 균형을 고려한 선택으로 추론한다.
- fine-tuning PNG는 내부 메타데이터가 없고 파일명상 `0.5drop_rate_finetune`으로 보인다. 따라서 그래프의 loss/accuracy 추세는 확정적으로 해석하되, 정확한 하이퍼파라미터는 확정하지 않는다.

## 4. 커밋 이력으로 본 개발 흐름

### 4.1 2026-05-26부터 2026-05-27: BPE tokenizer 기반 구축

초기 커밋 이후 `src/bpe.py`를 중심으로 byte-level BPE tokenizer를 구현했다.

핵심 커밋:

| 날짜 | 커밋 | 의미 |
| --- | --- | --- |
| 2026-05-26 | `ad5493a feat: init_special_tokens, train` | 특수 토큰과 BPE train loop 시작 |
| 2026-05-27 | `95ddcdd fix: correct BPE training merge loop` | merge loop 오류 수정 |
| 2026-05-27 | `0596abc`, `091051e` | BPE merge rule 저장/로드 |
| 2026-05-27 | `9ef2dbd`, `c38f78f` | BPE encode/decode 구현 |
| 2026-05-27 | `c200f54`, `1b6d1b3`, `051f468` | Colab/Drive에서 tokenizer 학습을 이어서 할 수 있게 개선 |

이 단계의 의도는 명확하다. 한국어 NSMC 리뷰를 외부 tokenizer 없이 다뤄야 하므로, 문자열 단위가 아니라 UTF-8 byte 단위에서 BPE를 직접 구현했다. `data/bpe_tokenizer_vocab3000.json`은 merge rule이 2740개이고, 기본 byte token 256개와 특수 토큰 4개를 합쳐 최종 vocab 3000을 구성한다.

### 4.2 2026-06-01: GPT core component 구현

`src/dataset.py`, `src/embeddings.py`, `src/attention.py`, `src/model.py`가 순서대로 구현됐다.

핵심 커밋:

| 커밋 | 구현 내용 |
| --- | --- |
| `c928131 feat: dataset` | token id를 next-token prediction input/target window로 자르는 Dataset |
| `7a6275b feat: embeddings` | token embedding + position embedding |
| `3a7e268 feat: attention` | causal multi-head self-attention |
| `891b8ca`, `6a2442e`, `c4678d7` | LayerNorm, GELU, FeedForward |
| `098765b`, `653ce95` | TransformerBlock |
| `9c72f5f`, `3eab6bf`, `7941e7e` | GPTModel init/forward와 후속 수정 |
| `c0ba6cc`, `e113e71` | greedy generation과 파라미터 수정 |

이 단계는 성능 개선보다는 “학습 가능한 모델을 정확히 만드는 단계”다. 이후 loss가 의미 있게 비교되려면 attention mask, positional embedding, cross entropy shape, generation sampling이 먼저 안정적이어야 했다.

### 4.3 2026-06-02: 학습 루프, checkpoint, generation, Colab 실행 안정화

`src/train.py`와 `gpt-lab.ipynb` 중심으로 pretraining 실행 루프가 만들어졌다.

핵심 커밋:

| 커밋 | 구현 내용 |
| --- | --- |
| `6736736 feat: calc_loss_batch` | batch 단위 next-token cross entropy |
| `5bdd917 feat: calc_loss_loader` | train/validation loader 평균 loss |
| `e6cc39b feat: save, load checkpoint` | checkpoint 저장/복원 |
| `d7910f7`, `42826ab` | temperature/top-k generation과 sampling 버그 수정 |
| `eb786e2 feat: generate_and_print_sample` | epoch별 생성 샘플 확인 |
| `87bb68c feat: train_model` | 실제 pretraining loop |
| `55a9779 fix: create attention mask on model device` | CUDA/MPS device mismatch 방지 |
| `a84fa72 fix: tolerate invalid utf8 in generated samples` | decode 실패를 `errors="replace"`로 완화 |
| `5493108 feat: cache pretraining token ids in notebook` | tokenization 반복 비용 절감 |

이 시점부터 팀은 “구현 완료”가 아니라 “학습이 왜 잘 안 내려가는지”를 보기 시작했다. 2026-06-02 이미지들은 대부분 `vocab_size=10000`, `emb=128`, `heads=4`, `layers=2`, `context=64`, `dropout=0.1`, `lr=0.0003` 조건이다. 2000 step까지 가도 validation loss가 약 `8.116`에 머물렀고, 이것이 vocab/token 희소성 가설로 이어졌다.

### 4.4 2026-06-03: 실험 관리, 하이퍼파라미터 탐색, fine-tuning

2026-06-03에는 실험 루프가 본격화됐다.

핵심 커밋:

| 커밋 | 구현 내용 |
| --- | --- |
| `7ba10c7 feat: implement sentiment finetuning` | GPT backbone 위 sequence classifier 구현 |
| `8995a14 feat: add notebook experiment controls` | 노트북에서 실험 변수 조정 가능 |
| `01540df data: bpe_tokenizer_vocab3000` | vocab 3000 tokenizer 추가 |
| `5dfcdbe fix: use data directory for notebook artifacts` | notebook artifact 경로 정리 |
| `80450af docs: add experiment plan templates` | EXP-001부터 EXP-006 실험 템플릿 추가 |
| `e8bb995 fix: reuse token cache across hyperparameters` | 하이퍼파라미터가 바뀌어도 token cache 재사용 |
| `452ba58`, `34ed758` | loss 그래프에 실험 조건 메타데이터 추가, 다운로드 기능 추가 |
| `19d763e feat: 최적의 하이퍼파라미터 설정` | 최종 notebook 기본값 반영 |
| `1671ecf chore: 모든 출력 지우기` | 제출/공유용으로 notebook 출력 제거 |

실험 관리 측면에서 가장 중요한 개선은 `80450af`, `e8bb995`, `452ba58`, `34ed758`이다. 실험 문서를 만들고, tokenization cache를 재사용하고, 결과 이미지에 `vocab_size`, `chars`, `tokens`, `context`, `batch`, `lr`, `weight_decay`, `model size`, `final loss`를 적기 시작했다. 이 덕분에 이후 그래프가 단순 스크린샷이 아니라 실험 로그로 기능했다.

## 5. 학습 시스템 구조

### 5.1 데이터

데이터는 NSMC 리뷰를 사용했다.

- 원본: `ratings_train.txt`, `ratings_test.txt`
- LM pretraining용:
  - `data/nsmc_lm_train.txt`
  - `data/nsmc_lm_val.txt`
- sentiment classification용:
  - `data/nsmc_sentiment_train.jsonl`
  - `data/nsmc_sentiment_val.jsonl`
  - `data/nsmc_sentiment_test.jsonl`

`download_data.py`는 다음 처리를 한다.

- 빈 리뷰 제거
- 공백 정규화
- train rows shuffle
- validation split 비율 `0.08`
- LM corpus는 문자 수 제한 기반으로 train/val text 생성
- sentiment dataset은 JSONL로 저장

### 5.2 Tokenizer

`src/bpe.py`의 tokenizer는 UTF-8 byte-level BPE다.

| 항목 | 값 |
| --- | --- |
| 특수 토큰 | `<pad>`, `<unk>`, `<bos>`, `<eos>` |
| 특수 토큰 ID | `0`, `1`, `2`, `3` |
| byte token 범위 | `4`부터 `259` |
| vocab 3000 구성 | 4 special + 256 byte + 2740 merge |
| encode | UTF-8 bytes -> merge 적용 -> token ids |
| decode | merge token을 재귀적으로 byte까지 펼친 뒤 UTF-8 decode |
| decode 안정화 | invalid UTF-8은 replacement character로 대체 |

한국어는 한 글자가 여러 byte로 구성되므로, 문자 단위 naive tokenization을 피하고 byte-level로 처리한 점이 중요하다. 또한 sentiment fine-tuning에서는 `encode(..., add_bos_eos=True)`를 사용해 문장 경계를 명시했다.

### 5.3 Pretraining Dataset

`src/dataset.py`는 token id sequence를 다음 토큰 예측용 window로 자른다.

- input: `token_ids[i : i + context_length]`
- target: `token_ids[i + 1 : i + context_length + 1]`
- stride 기본값: `context_length`
- 실험에서는 대부분 `TRAIN_STRIDE = CONTEXT_LENGTH`

중요한 관찰은 `context_length`를 키우면 한 sample이 길어지고 stride도 커져서 같은 token 수에서 update step 수가 줄어든다는 점이다. 이 때문에 context 128은 문맥 길이는 늘지만 epoch당 학습 업데이트가 줄어, 같은 epoch 수로는 불리했다.

### 5.4 GPT Model

모델 구조는 전형적인 decoder-only GPT다.

```text
token embedding + position embedding
-> dropout
-> N x TransformerBlock
   -> LayerNorm
   -> causal multi-head self-attention
   -> residual
   -> LayerNorm
   -> FeedForward
   -> residual
-> final LayerNorm
-> LM head
```

주요 구현 포인트:

- `MultiHeadAttention`은 Q/K/V projection, head split, causal mask, attention dropout, output projection을 수행한다.
- attention mask는 `attn_scores.device`에 생성하도록 고쳐 CUDA/MPS device mismatch를 막았다.
- `GPTModel.forward`는 `(batch, seq, vocab)` logits를 반환하고, targets가 있으면 flattened cross entropy를 계산한다.
- `generate`는 temperature와 top-k sampling을 지원한다.

### 5.5 Pretraining Loop

학습 루프는 다음 지표를 남긴다.

- epoch별 train loss
- epoch별 validation loss
- final held-out validation loss
- 같은 prompt `"이 영화는"`에 대한 생성 샘플
- PNG loss plot
- PNG 내부 실험 조건 메타데이터

노트북의 핵심 설정 셀은 최종적으로 다음 값을 갖는다.

| 변수 | 최종 notebook 기본값 |
| --- | --- |
| `TRAIN_CHARS` | `1_500_000` |
| `VAL_CHARS` | `100_000` |
| `NUM_EPOCHS` | `8` |
| `CONTEXT_LENGTH` | `16` |
| `BATCH_SIZE` | CUDA면 `16`, 아니면 `4` |
| `PRETRAIN_EMB_DIM` | `256` |
| `PRETRAIN_N_HEADS` | `8` |
| `PRETRAIN_N_LAYERS` | `4` |
| `PRETRAIN_DROP_RATE` | `0.0` |
| `PRETRAIN_LR` | `2e-3` |
| `PRETRAIN_WEIGHT_DECAY` | `0.1` |

### 5.6 Sentiment Fine-tuning

`src/finetune.py`는 GPT backbone에 classification head를 붙인다.

처리 방식:

- 리뷰 text를 BPE로 encode하고 BOS/EOS 추가
- `max_length`까지 truncate/pad
- GPT hidden state 계산
- pad가 아닌 마지막 token 위치의 hidden state를 sentence representation으로 사용
- dropout 후 linear classifier
- cross entropy loss와 accuracy 평가

기본 fine-tuning notebook 설정:

| 변수 | 값 |
| --- | --- |
| `SENTIMENT_TRAIN_LIMIT` | `20_000` |
| `SENTIMENT_VAL_LIMIT` | `5_000` |
| `SENTIMENT_TEST_LIMIT` | `5_000` |
| `SENTIMENT_MAX_LENGTH` | `128` |
| `FINETUNE_BATCH_SIZE` | CUDA면 `16`, 아니면 `4` |
| `FINETUNE_NUM_EPOCHS` | `3` |
| `FINETUNE_LR` | `1e-4` |
| `FINETUNE_WEIGHT_DECAY` | `0.01` |
| `FREEZE_BACKBONE` | `False` |

## 6. 팀의 성능 개선 루프

팀의 실험 방식은 다음 패턴으로 정리된다.

```text
1. 현상 관찰
   loss가 안 내려감, 생성 문장이 반복됨, validation gap이 생김

2. 병목 가설 설정
   vocab이 너무 큰가?
   learning rate가 부적절한가?
   context가 너무 짧거나 긴가?
   모델 capacity가 부족한가?
   dropout/weight decay가 과한가?
   리뷰 단위 문장 경계가 없어 next-token target이 불필요하게 어려운가?
   fine-tuning이 backbone을 잘 쓰고 있는가?

3. 한두 개 변수만 변경
   vocab_size, lr, context_length, emb_dim/layers/heads, drop_rate/weight_decay, BOS/EOS 경계 토큰 등

4. 동일 지표로 평가
   train loss, validation loss, train-val gap, 생성 샘플, fine-tuning accuracy

5. 결과 해석
   underfitting, overfitting, optimization instability, update step 감소, token sparsity 판단

6. 다음 기본값으로 반영
   notebook config 수정, 실험 문서 작성, 그래프 저장, token cache 재사용
```

이 루프의 강점은 loss만 보지 않고 `train loss와 val loss의 차이`, `생성 샘플`, `메타데이터가 붙은 그래프`, `다음 실험 계획`까지 함께 남겼다는 점이다.

## 7. 실험별 상세 분석

## 7.1 실험 0: 초기 smoke test와 baseline 진단

### 관찰

초기 2026-06-02 이미지들은 `vocab_size=10000`, `emb=128`, `heads=4`, `layers=2`, `dropout=0.1`, `context=64`, `lr=0.0003` 계열이었다. loss는 감소했지만 `8`대 초반에서 매우 느리게 움직였다.

| 이미지 | 주요 조건 | 최종 train | 최종 val | 해석 |
| --- | --- | ---: | ---: | --- |
| `loss_20260602_160221_steps86.png` | 100k train chars, 10k val chars, 86 steps | 8.312 | 8.411 | 짧은 smoke test, loss는 내려가지만 높음 |
| `loss_20260602_160907_steps200.png` | 100k train chars, 10k val chars, 200 steps | 8.136 | 8.414 | train만 내려가고 val 정체, 작은 corpus 과적합 신호 |
| `loss_20260602_163511_steps200.png` | 200k train chars, 20k val chars, 200 steps | 8.322 | 8.339 | corpus 증가로 train-val gap 완화 |
| `loss_20260602_164059_steps200.png` | 2M train chars, 200k val chars, 200 steps | 8.397 | 8.281 | 큰 corpus지만 update 부족, 아직 undertrained |
| `loss_20260602_172403_steps2000.png` | 2M train chars, 200k val chars, 2000 steps | 8.120 | 8.116 | 긴 학습에도 loss 개선 제한 |

### 가설

현재 corpus와 모델 크기에 비해 `vocab_size=10000`이 너무 크다. token 종류가 많으면 각 token의 관측 빈도가 줄고, 작은 모델이 next-token distribution을 학습하기 어렵다.

### 결론

이 baseline은 이후 vocab size를 줄이는 실험의 출발점이 되었다. 특히 `val loss`가 `8.1` 근처에서 머무는 현상이 “왜 안 떨어질까”라는 문제의식으로 이어졌다.

## 7.2 EXP-001: vocab size 10000 -> 3000

### 가설

`vocab_size=10000`은 현재 NSMC corpus와 작은 GPT model capacity에 비해 과하다. `vocab_size=3000`으로 줄이면 token sparsity가 줄고, loss가 더 빠르게 내려간다.

### 근거

- `data/bpe_tokenizer_vocab3000.json`이 추가됐다.
- 실제 파일은 merge 2740개를 포함해 vocab 3000을 구성한다.
- 2026-06-03 이후 `vocab_size=3000` 실험군에서는 같은 `emb=128, layers=2`에서도 validation loss가 7대에서 5대까지 내려갔다.
- `loss_20260603_133037_steps168 (1).png`는 `vocab_size=3000`, `train=100000 chars`, `val=20000 chars`, `context=64`, `lr=0.0003`, `emb=128`, `layers=2` 조건에서 3 epoch 후 `train=7.291`, `val=7.319`를 기록했다. 이는 2026-06-02의 10000 vocab baseline보다 loss scale이 낮다.

### 해석

vocab 3000으로 줄이면서 token 예측 문제가 쉬워졌다. 다만 이 실험은 완벽한 one-variable comparison은 아니다. 2026-06-02 baseline과 2026-06-03 실험은 train/val chars, epoch 수, eval 설정이 일부 다르다. 따라서 “vocab 3000이 항상 우월하다”보다는 “현재 작은 모델과 NSMC corpus에서 10000 vocab은 너무 어려웠고, 3000 vocab으로 바꾼 뒤 최적화가 훨씬 실용적인 구간으로 들어왔다”가 더 정확한 결론이다.

### 발표용 메시지

첫 번째 병목은 tokenizer였다. 모델 구조를 키우기 전에 tokenization 난도를 낮춰야 했다. vocab 3000은 작은 GPT가 학습 가능한 target distribution을 만드는 데 필요한 전제였다.

## 7.3 EXP-002: learning rate sweep

### 가설

`learning_rate`가 너무 낮으면 안정적이지만 느리게 학습하고, 너무 높으면 validation loss가 악화된다. 여러 값을 비교해 현재 model/data 조건에서 가장 낮은 validation loss를 찾는다.

### 고정 조건

| 항목 | 값 |
| --- | --- |
| train chars | 1,500,000 |
| val chars | 100,000 |
| vocab size | 3000 |
| model | emb=128, heads=4, layers=2 |
| context length | 64 |
| batch size | 16 |
| optimizer | AdamW |
| weight decay | 0.1 |

### 결과

| learning rate | epoch | final train loss | final val loss | 판단 |
| ---: | ---: | ---: | ---: | --- |
| 0.0001 | 8 | 6.415 | 6.394 | 너무 느림 |
| 0.0005 | 8 | 5.357 | 5.552 | 개선되지만 15 epoch 비교군보다 불리 |
| 0.001 | 15 | 4.980 | 5.312 | 안정적 |
| 0.002 | 15 | 4.998 | 5.249 | 최저 validation loss |
| 0.003 | 15 | 5.077 | 5.262 | 0.002와 근접하나 약간 나쁨 |
| 0.005 | 15 | 5.224 | 5.342 | 성능 악화 |
| 0.007 | 15 | 5.320 | 5.443 | 성능 악화 |
| 0.01 | 15 | 5.466 | 5.518 | 뚜렷한 악화 |

### 해석

`0.001 -> 0.002`에서 validation loss가 `5.312 -> 5.249`로 약 `0.063` 낮아졌다. 반면 `0.003` 이후에는 개선이 멈추고, `0.005`, `0.007`, `0.01`에서는 train과 validation loss가 모두 나빠졌다. 이는 너무 큰 learning rate가 안정적인 수렴을 방해했다는 신호다.

### 결론

`PRETRAIN_LR=0.002`를 채택했다. 이 값은 속도와 안정성의 균형점으로 보이며, 이후 context/model/regularization 실험의 기준값이 됐다.

## 7.4 EXP-003: context length 비교

### 가설

`context_length=64`는 긴 리뷰의 문맥을 잘라 token 예측 정보를 잃을 수 있다. `context_length=128`로 늘리면 더 긴 문맥을 학습해 validation loss와 생성 품질이 좋아질 것이다.

### 결과

외부 문서에는 `64 -> 128`뿐 아니라 `32`, `16`, `8` 결과도 함께 기록되어 있다.

| context length | final train loss | final val loss | 해석 |
| ---: | ---: | ---: | --- |
| 128 | 5.760 | 5.875 | baseline 64보다 나쁨 |
| 64 | 5.541 | 5.680 | 기준 |
| 32 | 5.378 | 5.532 | 더 짧게 하자 loss 개선 |
| 16 | 5.235 | 5.391 | loss 추가 개선 |
| 8 | 5.197 | 5.316 | loss만 보면 최저 |

### 왜 긴 context가 실패했나

`context_length`를 키우면 한 sample이 길어지고, 현재 dataset 구현은 stride도 context length와 같게 둔다. 따라서 같은 token 수에서 sample 수와 update step 수가 줄어든다.

실제 context 128 이미지:

- train tokens: 805,021
- val tokens: 58,388
- batch: 16
- context: 128
- actual steps: 3144
- final train loss: 5.760
- final val loss: 5.875

반면 context 64의 comparable run은 actual steps가 6288 또는 11790까지 가는 실험군이 있다. 같은 epoch 수로 비교하면 context 128은 업데이트 횟수에서 불리했다.

### 해석

가설은 현재 조건에서는 지지되지 않았다. 긴 문맥 자체가 무의미한 것은 아니지만, 현재 model capacity와 학습 step 수에서는 긴 문맥을 활용하기 전에 optimization이 부족했다. 오히려 짧은 context는 더 많은 update를 만들고, 짧은 NSMC 리뷰에서는 loss 기준으로 유리했다.

### 최종 설정에 대한 추론

loss만 보면 context 8이 가장 낮다. 그러나 notebook 최종 커밋은 `CONTEXT_LENGTH=16`을 선택했다. 이는 context 8의 loss 이점과 너무 짧은 문맥으로 인한 생성 품질 저하 가능성 사이에서, 최소한의 문맥을 유지하는 절충으로 보인다. 이 부분은 문서에 명시된 결론이 아니라, 커밋과 실험 결과를 함께 본 추론이다.

## 7.5 EXP-004: model capacity 확장

### 가설

`emb_dim=128`, `n_layers=2` 모델은 corpus와 tokenizer를 충분히 학습하기에는 capacity가 부족하다. model size를 키우면 train loss가 낮아지고, 과적합이 심하지 않다면 validation loss도 낮아진다.

### 결과

| model setting | batch | final train loss | final val loss | 해석 |
| --- | ---: | ---: | ---: | --- |
| emb=128, heads=4, layers=2 | 16 | 기준값 문서상 baseline | 약 5.68 계열 | 작은 모델 |
| emb=192, heads=4, layers=4 | 16 또는 8 | 5.189 | 5.462 | capacity 증가 효과 |
| emb=256, heads=8, layers=4 | 8 | 4.833 | 5.370 | validation 기준 best |

### 해석

모델을 키우면 train loss가 확실히 낮아졌다. `256/8/4`는 validation loss도 `192/4/4`보다 약 `0.092` 낮았다. 따라서 capacity 부족 가설은 지지된다.

다만 `256/8/4`에서는 train loss가 `4.833`, validation loss가 `5.370`으로 gap이 약 `0.537`까지 커졌다. 이는 모델이 학습 데이터를 더 잘 외우기 시작했고, validation 개선은 후반부에 정체됐다는 뜻이다.

### 결론

validation loss 기준으로는 `emb=256, heads=8, layers=4`를 채택할 만하다. 그러나 이후에는 early stopping, dropout/weight decay, learning rate schedule을 함께 봐야 한다.

## 7.6 EXP-005: dropout / weight decay 조정

### 가설

capacity를 키우면 overfitting 가능성이 커진다. dropout과 weight decay를 조정하면 train-val gap을 줄이고 validation loss를 낮출 수 있다.

### 결과

| drop rate | weight decay | final train loss | final val loss | 판단 |
| ---: | ---: | ---: | ---: | --- |
| 0.1 | 0.1 | 5.541 | 5.680 | baseline |
| 0.1 | 0.01 | 5.550 | 5.703 | weight decay 감소 효과 없음 |
| 0.1 | 0.05 | 5.547 | 5.695 | weight decay 감소 효과 없음 |
| 0.0 | 0.1 | 5.281 | 5.505 | best, gap 0.224 |
| 0.2 | best 기록 조건 | 5.709 | 5.802 | regularization 과함 |

### 해석

현재 설정에서는 dropout을 넣는 것이 일반화에 도움을 주기보다 학습 자체를 방해했다. `drop_rate=0.0`은 train loss와 val loss를 모두 낮췄고, train-val gap도 `0.224`로 성공 기준인 `0.3` 이하였다.

반면 `drop_rate=0.2`는 train loss와 validation loss가 모두 높아져 underfitting 경향을 보였다. weight decay를 `0.1`에서 `0.01` 또는 `0.05`로 낮추는 것도 validation loss 개선으로 이어지지 않았다.

### 결론

`PRETRAIN_DROP_RATE=0.0`, `PRETRAIN_WEIGHT_DECAY=0.1`을 채택했다. 최종 notebook도 이 결정을 반영한다.

## 7.7 EXP-006: sentiment fine-tuning

### 가설

pretraining loss가 낮아져도 sentiment classification 성능이 자동으로 좋아지는 것은 아니다. backbone freeze 여부, fine-tuning learning rate, max_length, train data size가 validation/test accuracy를 좌우한다.

### 구현

`src/finetune.py`에서 다음 구조를 만들었다.

```text
review text
-> BPE encode with BOS/EOS
-> pad/truncate
-> GPT backbone hidden states
-> last non-pad token hidden state
-> dropout
-> linear classifier
-> cross entropy
```

### 관찰된 fine-tuning 그래프

`0.5drop_rate_finetune.png` 기준:

| epoch | train loss | val loss | train acc | val acc |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 약 0.255 | 약 0.545 | 약 0.897 | 약 0.773 |
| 2 | 약 0.150 | 약 0.695 | 약 0.945 | 약 0.775 |
| 3 | 약 0.090 | 약 0.845 | 약 0.968 | 약 0.770 |

### 해석

fine-tuning은 명확한 overfitting을 보인다.

- train loss는 계속 감소한다.
- train accuracy는 약 0.97까지 오른다.
- validation accuracy는 약 0.77 근처에서 정체된다.
- validation loss는 오히려 상승한다.

즉, classifier 또는 backbone이 train subset에는 빠르게 맞춰지지만 validation generalization은 개선되지 않았다.

### 다음 개선 방향

- `FREEZE_BACKBONE=True`로 classifier만 학습해 비교
- `FINETUNE_LR=5e-5`처럼 더 작은 learning rate 실험
- validation loss가 최소인 epoch 1 또는 epoch 2 checkpoint 사용
- train limit을 20k에서 80k로 늘려 데이터 효과 확인
- class imbalance와 confusion examples 확인
- max_length 128 vs 256 비교

## 7.8 EXP-007: BOS/EOS 문장 경계 토큰

### 배경

이 실험은 앞선 하이퍼파라미터 튜닝보다 “데이터를 GPT가 풀기 쉬운 next-token 문제로 바꾸는” 성격이 강하다. NSMC는 원래 리뷰 단위 데이터인데, pretraining용 LM corpus로 단순 연결하면 한 리뷰의 마지막 token 다음에 전혀 다른 리뷰의 첫 token이 바로 이어진다. 모델 입장에서는 문장이나 리뷰가 끝났다는 신호 없이, 의미적으로 불연속인 다음 리뷰 시작 token까지 예측해야 한다.

`<bos>`와 `<eos>`는 이 문제를 직접 겨냥한다.

- `<bos>`는 “새 리뷰/문장이 시작된다”는 조건을 제공한다.
- `<eos>`는 “여기서 리뷰/문장이 끝난다”는 target을 제공한다.
- 생성 단계에서는 `<eos>`가 나오면 출력을 멈출 수 있어 무의미한 반복을 줄일 수 있다.
- sentiment fine-tuning에서는 마지막 non-pad token이 `<eos>`가 되므로, classifier가 “문장 전체를 본 뒤의 종료 위치 hidden state”를 사용하게 된다.

### 커밋과 코드 근거

| 근거 | 확인한 내용 |
| --- | --- |
| `ad5493a feat: init_special_tokens, train` | 2026-05-26에 `<pad>`, `<unk>`, `<bos>`, `<eos>`를 고정 ID `0~3`으로 초기화하는 기반이 들어갔다. |
| `9ef2dbd feat: implement BPE encode` | 2026-05-27에 `encode(text, add_bos_eos=True)`가 구현되어, 필요 시 앞뒤로 BOS/EOS를 붙일 수 있게 됐다. |
| `42826ab fix: correct token sampling in generate` | 2026-06-02에 `generate(..., eos_id=...)`가 `eos_id`를 만나면 생성을 중단하도록 수정됐다. 즉 EOS는 학습 token뿐 아니라 생성 종료 조건으로도 설계됐다. |
| `7ba10c7 feat: implement sentiment finetuning` | 2026-06-03에 `ReviewSentimentDataset`이 `tokenizer.encode(row["text"], add_bos_eos=True)`를 사용하도록 구현됐다. |
| 외부 문서 `EOS / BOS` | 2026-06-03 15:35 생성된 문서에 “넣었어요. 코드 확인 해주세요. jinhyuk branch에 있음.”이라고 기록되어 있다. |

현재 HEAD의 notebook pretraining token cache 셀에는 `train_token_ids = tokenizer.encode(train_text)`, `val_token_ids = tokenizer.encode(val_text)` 형태가 남아 있다. 따라서 저장된 코드만 보면 pretraining corpus 전체에 BOS/EOS를 리뷰 단위로 붙인 최종 run은 완전히 보존되어 있지 않다. 다만 tokenizer, generate, fine-tuning 코드는 모두 BOS/EOS를 지원하고, 팀 회고상 이 변경이 마지막 성능 개선으로 기록된다.

### 가설

문장 경계 token을 넣으면 validation loss가 크게 낮아진다. 이유는 다음과 같다.

1. 리뷰 사이의 불연속 전이를 억지로 학습하지 않아도 된다.
2. 모델이 “문장 시작 분포”와 “문장 내부 분포”를 분리해서 볼 수 있다.
3. 리뷰 끝에서는 다음 리뷰의 첫 token 대신 `<eos>`라는 명확한 target을 예측한다.
4. 짧은 NSMC 리뷰에서는 경계 token의 빈도가 충분히 높아서, 작은 GPT도 종료 패턴을 빠르게 학습한다.
5. 이후 생성/분류 task에서도 같은 경계 신호를 재사용할 수 있다.

### 의도된 구현 방식

pretraining에서 효과를 보려면 corpus 전체에 한 번만 BOS/EOS를 붙이는 것이 아니라, 리뷰 단위로 붙여야 한다.

```text
review_1 -> <bos> review_1 <eos>
review_2 -> <bos> review_2 <eos>
review_3 -> <bos> review_3 <eos>
...
```

구현 형태는 다음과 같은 구조가 가장 일관적이다.

```python
token_ids = []
for review in reviews:
    token_ids.extend(tokenizer.encode(review, add_bos_eos=True))
```

또 한 가지 중요한 재현 포인트가 있다. notebook은 tokenization 결과를 `data/cache`에 저장하고 재사용한다. BOS/EOS 적용 여부를 바꿀 때는 기존 token cache를 삭제하거나, cache key에 `add_bos_eos_for_pretrain=True` 같은 정책 값을 반드시 포함해야 한다. 그렇지 않으면 “BOS/EOS를 켰다고 생각했지만 기존 non-BOS/EOS token cache를 재사용하는” 실험 오염이 생긴다.

### 결과와 해석

첨부된 실제 BOS/EOS run 그래프에서는 final validation loss가 `4.745`까지 내려갔다. 팀이 기억하는 개선 흐름은 BOS/EOS를 마지막에 추가한 뒤 validation loss가 약 `5.05 -> 4.7` 수준으로 떨어진 것이다. 절대 loss 개선폭은 약 `0.35`이며, perplexity 기준으로는 `exp(5.05) ~= 156`에서 `exp(4.70) ~= 110`으로 줄어든 것과 같아 약 `30%` 수준의 감소다.

![BOS/EOS actual validation loss 4.745](assets/gpt-training-analysis/bos_eos_actual_val4745.png)

이 개선폭은 앞선 단일 하이퍼파라미터 조정보다 크다.

| 변경 | 대표 validation loss 변화 | 해석 |
| --- | ---: | --- |
| learning rate `0.001 -> 0.002` | `5.312 -> 5.249`, 약 `-0.063` | optimizer 균형점 탐색 |
| model capacity `192/4/4 -> 256/8/4` | `5.462 -> 5.370`, 약 `-0.092` | 표현력 증가 |
| dropout 제거 계열 | `5.680 -> 5.505` 계열 | underfitting 완화 |
| BOS/EOS 경계 token 추가 | 실제 run final val `4.745`, 팀 회고상 `5.05 -> 4.7` | 데이터의 next-token 구조 자체를 개선 |

따라서 BOS/EOS는 단순한 특수 token 추가라기보다, “리뷰 단위 corpus를 문장 경계가 있는 언어 모델링 문제로 재정의한 변경”으로 해석하는 것이 맞다. 모델 크기와 learning rate를 많이 조정해도 남아 있던 병목이, 사실상 데이터 경계 정보 부재였을 가능성이 높다.

### 발표용 메시지

마지막 개선은 모델을 더 키운 것이 아니라, 모델에게 문제의 구조를 알려준 것이다. NSMC 리뷰는 독립된 짧은 문장들의 집합인데, BOS/EOS가 없으면 GPT는 리뷰와 리뷰 사이의 무작위 연결까지 next-token으로 배워야 했다. 경계 token을 넣자 loss가 약 `5.05`에서 `4.7`까지 떨어졌고, 이는 우리 실험 루프에서 가장 극적인 후반부 개선이었다.

### Warmup

`웜업` 문서는 생성만 되어 있고 결과는 없다. 현재 notebook과 `src/train.py`에는 warmup이나 cosine decay가 본격 적용된 흔적은 없다. README의 추가 미션에 warmup/cosine decay/gradient clipping이 제안되어 있으므로, 다음 개선 후보로 남아 있다.

## 8. 최종 하이퍼파라미터 선택의 근거

최종 커밋 `19d763e`의 notebook 설정:

| 변수 | 최종값 | 근거 |
| --- | ---: | --- |
| `vocab_size` | 3000 | 10000 vocab의 loss 정체와 token sparsity 문제 |
| `CONTEXT_LENGTH` | 16 | context sweep에서 짧은 context가 loss 개선. 8보다 문맥을 조금 더 보존하는 절충으로 추정 |
| `PRETRAIN_EMB_DIM` | 256 | capacity 실험에서 256/8/4가 validation 기준 best |
| `PRETRAIN_N_HEADS` | 8 | 256-dim model과 함께 사용 |
| `PRETRAIN_N_LAYERS` | 4 | capacity 확장 효과 |
| `PRETRAIN_DROP_RATE` | 0.0 | dropout 제거 시 validation loss 개선 |
| `PRETRAIN_LR` | 0.002 | learning rate sweep 최저 validation loss |
| `PRETRAIN_WEIGHT_DECAY` | 0.1 | weight decay 감소 실험에서 개선 없음 |
| `TRAIN_CHARS` | 1,500,000 | 실험 템플릿의 표준 corpus size |
| `VAL_CHARS` | 100,000 | 실험 템플릿의 표준 validation size |
| `NUM_EPOCHS` | 8 | Colab T4에서 반복 가능한 학습 시간과 성능 균형 |

## 9. 성능 개선의 핵심 인사이트

### 9.1 먼저 tokenization 난도를 낮췄다

초기에는 vocab 10000에서 loss가 8대 초반에 정체됐다. 작은 GPT에서 너무 큰 vocab은 token distribution이 희소해지고, embedding/head parameter도 늘어나며, 각 token을 볼 기회가 줄어든다. vocab 3000으로 낮추자 이후 실험에서 loss가 5대까지 내려가는 기반이 만들어졌다.

### 9.2 learning rate가 가장 직접적인 최적화 변수였다

`0.0001`은 너무 느렸고, `0.005` 이상은 validation loss를 악화시켰다. `0.002`가 가장 낮은 validation loss를 기록했다. 이 실험은 가장 잘 통제된 sweep 중 하나이며, 최종 설정으로 반영되었다.

### 9.3 context length는 “길수록 좋다”가 아니었다

NSMC 리뷰처럼 짧은 텍스트에서는 긴 context가 항상 이득이 아니다. 특히 stride가 context length와 같으면 context가 길수록 update 수가 줄어든다. 이번 실험에서는 context 128이 baseline 64보다 나빴고, 8/16/32가 더 좋은 loss를 보였다.

### 9.4 모델 capacity는 효과가 있었지만 gap도 커졌다

`128/4/2`에서 `256/8/4`로 키우자 validation loss가 낮아졌다. 동시에 train-val gap이 커져 regularization/early stopping 필요성이 생겼다. 이것이 dropout/weight decay 실험으로 이어졌다.

### 9.5 현재 pretraining에서는 dropout이 과했다

dropout 0.2는 underfitting, dropout 0.1도 best가 아니었다. dropout 0.0에서 validation loss가 더 낮고 gap도 허용 범위였다. 작은 corpus와 작은 모델에서는 dropout보다 충분한 fitting 능력이 더 중요했다.

### 9.6 fine-tuning은 별도의 generalization 문제가 있다

pretraining 개선만으로 sentiment classification이 자동 개선되지는 않았다. fine-tuning 그래프는 train accuracy가 매우 높아지지만 validation accuracy가 정체되는 전형적 과적합 패턴이다.

## 10. 발표용 스토리라인

### Slide 1. 문제 정의

우리는 외부 pretrained model/tokenizer 없이 NSMC 기반 Mini GPT를 직접 구현하고, 학습 loss와 downstream sentiment 성능을 개선해야 했다.

### Slide 2. 구현 기반

UTF-8 byte-level BPE, GPTDataset, causal multi-head attention, GPTModel, train/eval/generate loop, sentiment classifier를 모두 직접 구현했다.

### Slide 3. 실험 루프

현상 관찰 -> 가설 설정 -> 한 변수 변경 -> train/val loss 측정 -> 생성 샘플 확인 -> 다음 설정 반영.

### Slide 4. 초기 문제

`vocab_size=10000`에서 2000 step을 학습해도 validation loss가 약 `8.116`에 머물렀다. 생성 문장도 조사/빈출 토큰 반복이 많았다.

### Slide 5. 가설 1: vocab size

10000 vocab은 작은 corpus/model에 과하다. vocab 3000으로 줄여 token sparsity를 낮췄다.

### Slide 6. 가설 2: learning rate

`0.0001`은 느리고, `0.005` 이상은 불안정했다. `0.002`가 validation loss `5.249`로 best였다.

### Slide 7. 가설 3: context length

긴 context가 무조건 좋지 않았다. context 128은 update 수 감소로 baseline보다 나빴고, 짧은 context가 loss 기준으로 유리했다. 최종적으로 16을 채택했다.

### Slide 8. 가설 4: model capacity

`128/4/2`는 capacity가 부족했다. `256/8/4`로 키우자 validation loss가 `5.370`까지 낮아졌다.

### Slide 9. 가설 5: regularization

dropout을 줄이거나 제거했을 때 validation loss가 개선됐다. 현재 조건에서는 dropout보다 모델의 학습 능력이 더 중요했다.

### Slide 10. Fine-tuning 결과

classification head를 붙여 sentiment fine-tuning을 수행했다. train accuracy는 약 0.97까지 올랐지만 validation accuracy는 약 0.77에 정체했고 val loss는 상승했다.

### Slide 11. 최종 설정

`vocab=3000`, `context=16`, `emb=256`, `heads=8`, `layers=4`, `dropout=0.0`, `lr=0.002`, `weight_decay=0.1`.

### Slide 12. 남은 개선 과제

warmup/cosine decay, gradient clipping, early stopping, repeated seed validation, freeze backbone fine-tuning, confusion example 분석.

## 11. 발표에서 강조할 문장

- “우리는 loss가 안 떨어지는 현상을 구현 버그로만 보지 않고, tokenizer 난도, optimization, context/update 수, capacity, regularization 문제로 쪼개서 검증했다.”
- “vocab 10000에서 loss가 8대에 머문 것은 모델이 언어를 못 배운다기보다, 작은 모델에게 너무 어려운 token prediction 문제를 준 결과로 해석했다.”
- “learning rate sweep은 가장 직접적인 개선이었다. 0.002에서 validation loss가 가장 낮았고, 0.005 이상은 오히려 악화됐다.”
- “context length는 길수록 좋은 것이 아니라, 주어진 데이터와 update budget 안에서 최적점이 있다.”
- “model capacity를 키우면 loss는 내려갔지만 train-val gap도 커져 다음 실험이 regularization으로 이어졌다.”
- “fine-tuning은 pretraining과 다른 문제였다. train accuracy가 높아졌지만 validation이 정체되어 downstream generalization 개선이 남아 있다.”

## 12. 실험 관리 측면 평가

잘한 점:

- 실험 템플릿을 만들고, 가설/고정 변수/변경 변수/성공 기준/결론을 기록하려 했다.
- 그래프에 실험 조건을 넣어 나중에 재분석 가능하게 만들었다.
- token cache를 재사용해 하이퍼파라미터 sweep 비용을 줄였다.
- train loss만 보지 않고 validation loss와 생성 샘플을 함께 봤다.
- final notebook config에 실험 결과를 반영했다.

보완할 점:

- seed를 모든 실험에 명시해야 한다.
- 각 실험의 baseline과 experiment가 완전히 같은 조건인지 더 엄격히 관리해야 한다.
- 일부 문서의 상태가 `시작 전` 또는 `진행 중`으로 남아 있고, 실제 결과와 싱크가 맞지 않는다.
- 생성 샘플 평가는 정성적이므로 반복률, unknown/replacement character 비율 같은 보조 지표를 추가하면 좋다.
- fine-tuning 결과 이미지는 메타데이터가 없어 정확한 설정 재현이 어렵다.
- context 16 채택 이유처럼 최종 의사결정 근거를 문서에 명시해야 한다.

## 13. 다음 실험 제안

### 13.1 Pretraining

1. `context_length=16`, `emb=256`, `heads=8`, `layers=4`, `dropout=0.0`, `lr=0.002` 설정으로 seed 3개 반복
2. validation loss 기준 early stopping 도입
3. warmup 5 percent + cosine decay 적용
4. gradient clipping 적용
5. `context_length=16`과 `context_length=32`를 같은 actual step 수로 재비교
6. `dropout=0.0`과 `dropout=0.05`를 큰 모델에서 재비교

### 13.2 Generation Quality

1. 동일 prompt 10개 고정
2. 반복 n-gram 비율 측정
3. replacement character 비율 측정
4. 평균 문장 길이와 EOS 도달률 측정
5. temperature/top-k grid search

### 13.3 Fine-tuning

1. `FREEZE_BACKBONE=True`와 `False` 비교
2. `FINETUNE_LR=5e-5`, `1e-4`, `3e-4` 비교
3. validation loss 최소 checkpoint 선택
4. train data 20k vs 80k 비교
5. max_length 128 vs 256 비교
6. 오답 예시를 긍정/부정/부정어/비꼼/짧은 리뷰로 분류

## 14. 부록 A: 직접 확인한 이미지 요약

아래 이미지들은 사용자가 지정한 원본 파일과 Notion export 하위 이미지를 프로젝트 내부 `assets/gpt-training-analysis/`로 복사한 뒤 상대 경로로 연결한 것이다. 이 문서를 Markdown preview로 열면 각 실험 그래프가 바로 렌더링된다.

### 이미지 갤러리: 초기 baseline / vocab size

| 실험 | 이미지 |
| --- | --- |
| 매우 짧은 smoke test: 다운로드 | ![Smoke test download](assets/gpt-training-analysis/smoke_download.png) |
| 매우 짧은 smoke test: loss가 왜 안 떨어지는가 | ![Smoke test why not decrease](assets/gpt-training-analysis/smoke_why_not_decrease.png) |
| 작은 corpus 20k/5k 실험 | ![Small corpus 20k train 5k val](assets/gpt-training-analysis/small_corpus_20k_5k.png) |
| 기본 설정 loss curve | ![Baseline basic loss curve](assets/gpt-training-analysis/baseline_basic.png) |
| MPS, emb=32, layer=1, dropout=0.0 pretraining | ![Pretraining MPS emb32 layers1 dropout0](assets/gpt-training-analysis/pretrain_mps_emb32_layers1_dropout0.png) |
| 168 steps run, metadata N/A | ![Pretraining 168 steps no metadata](assets/gpt-training-analysis/pretrain_steps168_no_metadata.png) |
| 168 steps run, train 100k / val 20k | ![Pretraining 168 steps 100k 20k](assets/gpt-training-analysis/pretrain_steps168_100k20k.png) |
| vocab 10000, train 100k / val 10k, 86 steps | ![Vocab 10000 100k 10k 86 steps](assets/gpt-training-analysis/vocab10000_100k10k_steps86.png) |
| vocab 10000, train 100k / val 10k, 200 steps | ![Vocab 10000 100k 10k 200 steps](assets/gpt-training-analysis/vocab10000_100k10k_steps200.png) |
| vocab 10000, train 200k / val 20k, 200 steps | ![Vocab 10000 200k 20k 200 steps](assets/gpt-training-analysis/vocab10000_200k20k_steps200.png) |
| vocab 10000, train 2M / val 200k, 200 steps | ![Vocab 10000 2M 200k 200 steps](assets/gpt-training-analysis/vocab10000_2m200k_steps200.png) |
| vocab 10000, train 2M / val 200k, 2000 steps | ![Vocab 10000 2M 200k 2000 steps](assets/gpt-training-analysis/vocab10000_2m200k_steps2000.png) |
| Notion export의 vocab 10000 baseline | ![Vocab 10000 baseline from Notion export](assets/gpt-training-analysis/vocab10000_baseline_notion.png) |
| vocab 3000, 8 epoch loss curve | ![Vocab 3000 8 epoch loss curve](assets/gpt-training-analysis/vocab3000_8epoch_curve.png) |

### 이미지 갤러리: learning rate sweep

| 실험 | 이미지 |
| --- | --- |
| lr=0.0001, 6288 steps | ![Learning rate 0.0001 6288 steps](assets/gpt-training-analysis/lr_0_0001_steps6288.png) |
| lr=0.0005, 6288 steps | ![Learning rate 0.0005 6288 steps](assets/gpt-training-analysis/lr_0_0005_steps6288.png) |
| lr=0.001, 6288 steps | ![Learning rate 0.001 6288 steps](assets/gpt-training-analysis/lr_0_001_steps6288.png) |
| lr=0.001, 11790 steps | ![Learning rate 0.001 11790 steps](assets/gpt-training-analysis/lr_0_001_steps11790.png) |
| lr=0.002, 11790 steps | ![Learning rate 0.002 11790 steps](assets/gpt-training-analysis/lr_0_002_steps11790.png) |
| lr=0.003, 11790 steps | ![Learning rate 0.003 11790 steps](assets/gpt-training-analysis/lr_0_003_steps11790.png) |
| lr=0.005, 11790 steps | ![Learning rate 0.005 11790 steps](assets/gpt-training-analysis/lr_0_005_steps11790.png) |
| lr=0.007, 11790 steps | ![Learning rate 0.007 11790 steps](assets/gpt-training-analysis/lr_0_007_steps11790.png) |
| lr=0.010, 11790 steps | ![Learning rate 0.010 11790 steps](assets/gpt-training-analysis/lr_0_010_steps11790.png) |

### 이미지 갤러리: context / model capacity / regularization / fine-tuning

| 실험 | 이미지 |
| --- | --- |
| context length 64 -> 128 | ![Context length 128 loss curve](assets/gpt-training-analysis/context128_steps3144.png) |
| model capacity: emb=192, heads=4, layers=4 | ![Model capacity emb192 heads4 layers4](assets/gpt-training-analysis/model_capacity_192_4_4.png) |
| model capacity: emb=256, heads=8, layers=4 | ![Model capacity emb256 heads8 layers4](assets/gpt-training-analysis/model_capacity_256_8_4.png) |
| regularization: dropout=0.2, weight_decay=0.01 | ![Regularization dropout 0.2 weight decay 0.01](assets/gpt-training-analysis/regularization_dropout02_wd001.png) |
| regularization: dropout=0.0, weight_decay=0.1 | ![Regularization dropout 0.0 weight decay 0.1](assets/gpt-training-analysis/regularization_dropout00_wd01.png) |
| BOS/EOS boundary token 적용 run | ![BOS EOS actual validation loss 4.745](assets/gpt-training-analysis/bos_eos_actual_val4745.png) |
| sentiment fine-tuning: dropout=0.5 | ![Sentiment fine-tuning dropout 0.5](assets/gpt-training-analysis/sentiment_finetune_dropout05.png) |

### 사용자가 직접 지정한 이미지

| 파일 | 확인한 내용 |
| --- | --- |
| `0.5drop_rate_finetune.png` | sentiment fine-tuning. train loss 감소, train acc 상승, val loss 상승, val acc 정체. 과적합 신호 |
| `image.png` | Mini GPT pretraining. 메타데이터 일부 N/A, mps device, emb=32, layers=1, dropout=0.0, final train=5.709, val=5.802 |
| `loss_20260603_154249_steps11790.png` | lr=0.007, final train=5.320, val=5.443 |
| `loss_20260603_133037_steps11790.png` | lr=0.001, final train=4.980, val=5.312 |
| `loss_20260603_133037_steps11790 (1).png` | lr=0.002, final train=4.998, val=5.249 |
| `loss_20260603_133037_steps11790 (2).png` | lr=0.003, final train=5.077, val=5.262 |
| `loss_20260603_133037_steps11790 (3).png` | lr=0.005, final train=5.224, val=5.342 |
| `loss_20260603_133037_steps11790 (4).png` | lr=0.01, final train=5.466, val=5.518 |
| `loss_20260603_133037_steps6288.png` | lr=0.0001, 8 epoch, final train=6.415, val=6.394 |
| `loss_20260603_133037_steps6288 (1).png` | lr=0.0005, 8 epoch, final train=5.357, val=5.552 |
| `loss_20260603_133037_steps6288 (2).png` | lr=0.001, 8 epoch, final train=5.211, val=5.432 |
| `loss_20260603_133037_steps168.png` | metadata N/A가 많지만 final train=7.291, val=7.319 |
| `loss_20260603_133037_steps168 (1).png` | train=100k chars, val=20k chars, context=64, lr=0.0003, final train=7.291, val=7.319 |
| `loss_20260602_172403_steps2000.png` | vocab_size=10000 baseline, 2M train chars, 200k val chars, final train=8.120, val=8.116 |
| `loss_20260602_164059_steps200.png` | 2M train chars, 200k val chars, 200 steps, final train=8.397, val=8.281 |
| `loss_20260602_163511_steps200.png` | 200k train chars, 20k val chars, final train=8.322, val=8.339 |
| `loss_20260602_160907_steps200.png` | 100k train chars, 10k val chars, final train=8.136, val=8.414 |
| `loss_20260602_160221_steps86.png` | 100k train chars, 10k val chars, final train=8.312, val=8.411 |
| `traincorpus20_000_val5000.png` | small corpus run. loss가 9.36대에서 8.3대 후반까지 감소 |
| `다운로드.png` | 매우 짧은 smoke test. global step 0, 5, 8만 기록 |
| `왜 안떨어질까.png` | 매우 짧은 smoke test. loss 감소 폭이 작아 초기 문제의식에 해당 |

### Notion export 하위 이미지

| 파일 | 확인한 내용 |
| --- | --- |
| `context length 64 -> 128/loss_20260603_142516_steps3144.png` | context=128, final train=5.760, val=5.875 |
| `model_capacity/image.png` | emb=192, heads=4, layers=4 계열. train history `[7.255, ..., 5.189]`, val history `[6.849, ..., 5.462]` |
| `model_capacity/image 1.png` | emb=256, heads=8, layers=4. final train=4.833, val=5.370 |
| `dropout weight_decay 조정/weight_decay0.01_drop_rate0.2.png` | metadata N/A가 많지만 final train=5.725, val=5.840로 강한 regularization이 불리 |
| `dropout weight_decay 조정/drop_rate0.0_weight_decay0.1.png` | final train=5.709, val=5.802로 표시. 문서 표의 drop_rate=0.0 best 기록과 일부 수치 차이가 있어 문서 표와 함께 해석 |
| `기본/image.png` | 기본 설정 loss curve. train history `[7.355, ..., 5.541]`, val history `[7.268, ..., 5.68]` |
| `Vocab size 10000 -> 3000/mini_gpt_pretraing_loss_-_yj.png` | 8 epoch mini GPT pretraining curve. 최종 loss가 5.5 근처까지 내려감 |
| `Vocab size 10000 -> 3000/loss_20260602_172403_steps2000.png` | vocab 10000 baseline과 동일 이미지, final val=8.116 |

## 15. 부록 B: 외부 실험 문서 상태 요약

| 문서 | 상태 | 핵심 내용 |
| --- | --- | --- |
| `Vocab size 10000 -> 3000` | 진행 중 | vocab 10000 loss 정체를 보고 vocab 3000 가설 수립 |
| `EXP-001 Vocab Size 비교 실험` | 시작 전 | 템플릿 성격, 일부 결과 수치 포함 |
| `learning_rate 비교 실험 보고서` | 시작 전으로 표기됐지만 결과 있음 | lr sweep 결과, `0.002` 채택 |
| `context length 64 -> 128` | 완료 | 128은 실패, 더 짧은 context가 loss 기준 우수 |
| `model_capacity` | 완료 | 256/8/4가 validation 기준 best, gap 증가 |
| `dropout weight_decay 조정` | 완료 | dropout 0.0 채택, dropout 0.2는 underfitting |
| `sentiment fine-tuning 설정 비교` | 시작 전 | 실험 설계만 문서화 |
| `EOS BOS` | 시작 전으로 표시됐지만 구현 메모 있음 | 2026-06-03 15:35에 BOS/EOS를 넣었다고 기록. 팀 회고상 마지막 run에서 val loss가 약 `5.05 -> 4.7`로 개선 |
| `웜업` | 시작 전 | 결과 없음 |
| `기본` | 시작 전 | 기본 설정 train/val history 기록 |

## 16. 최종 결론

우리 팀의 GPT 학습 개선 과정은 무작정 모델을 키운 것이 아니라, 학습이 막히는 지점을 단계적으로 분해한 과정이었다.

처음에는 직접 구현한 BPE와 GPT component가 정상 동작하도록 만들었다. 그 다음 `vocab_size=10000` 조건에서 loss가 잘 내려가지 않는 현상을 보고 token sparsity 가설을 세웠고, `vocab_size=3000`으로 학습 난도를 낮췄다. 이후 learning rate sweep을 통해 `0.002`를 채택했고, context length 실험을 통해 “긴 문맥이 항상 좋은 것은 아니다”라는 결론을 얻었다. 모델 capacity를 키워 validation loss를 더 낮췄지만 train-val gap이 커져 regularization 실험으로 이어졌고, 현재 조건에서는 dropout을 제거하는 편이 더 나았다. 마지막에는 BOS/EOS로 리뷰 단위 문장 경계를 명시하면서, 팀 회고상 validation loss가 약 `5.05`에서 `4.7`까지 떨어지는 가장 극적인 후반부 개선을 얻었다.

최종 pretraining 기본값은 이 실험들의 누적 결과다. 다만 sentiment fine-tuning은 아직 완성된 개선 루프가 아니라, 과적합 문제를 확인한 단계다. 발표에서는 pretraining loop는 “가설 기반으로 상당히 개선했다”고 말할 수 있고, fine-tuning은 “다음 실험 과제가 명확해졌다”고 정리하는 것이 정확하다.
