```
input_layer_6 (None, 24)
        │
  Embedding (vocab_q, 50)
        │
  ┌────────────────────────┐
  │ encoder_lstm_1         │
  │ (None, 24, 50), (None, 50), (None, 50) │
  └────────────────────────┘
         │ encoder_output_1
         ▼
    encoder_combined (24, 100)

────────────────────────────────────────────────────────────────────

input_layer_7 (None, 36)
         │
  Embedding (vocab_a, 50)
         │
  ┌────────────────────────┐       ┌────────────────────────┐
  │ decoder_lstm_1         │       │ decoder_lstm_2         │
  │ (None, 36, 50), (None, 50), (None, 50) │   (None, 36, 50), (None, 50), (None, 50) │
  └────────────────────────┘       └────────────────────────┘
         │ decoder_output_1                │ decoder_output_2
         └────────┬────────┘               └────────┬────────┘
                  ▼                           ▼
        Concatenate([decoder_output_1, decoder_output_2]) 
                  ▼
         decoder_combined (36, 100)

────────────────────────────────────────────────────────────────────

     decoder_combined ─┐
                        │
                        ▼
       AdditiveAttention([decoder_combined, encoder_combined]) 
                        ▼
               context_vector (36, 100)
                        ▼
              Dropout(0.2) → Dense(50) → Adjusted context_vector
                        ▼
  Concatenate([decoder_combined, context_vector]) 
                        ▼
         decoder_context_concat (36, 150) 
                        ▼
   TimeDistributed(Dense(vocab_size)) → Final Prediction (36, vocab_size)

```
1. 인코더:

 - input_layer_6 → Embedding → 1개의 LSTM (encoder_lstm_1) → encoder_combined(24, 100)

2. 디코더:

 - input_layer_7 → Embedding → 두 개의 병렬 LSTM (decoder_lstm_1, decoder_lstm_2)

  - decoder_lstm_1은 Embedding과 encoder_lstm_1의 출력을 입력으로 받습니다.

  - decoder_lstm_2는 Embedding과 decoder_lstm_1의 출력을 입력으로 받습니다.

 - 두 LSTM의 출력을 decoder_combined로 합칩니다.

3. 어텐션:

 - decoder_combined와 encoder_combined를 기반으로 어텐션을 적용하여 context_vector 생성, 이를 통해 최종 예측값을 도출합니다.
