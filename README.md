```
   ┌────────────────────┐
   │ encoder_input (24) │
   └──────────┬─────────┘
              ▼
      ┌───────────────────────┐
      │ Embedding (vocab_q, 50)│
      └──────────┬────────────┘
                 ▼
         ┌──────────────┐         ┌──────────────┐
         │ Encoder LSTM1│         │ Encoder LSTM2│
         └─────┬────────┘         └─────┬────────┘
               │ encoder_output_1       │ encoder_output_2
               └────────┬───────────────┘
                        ▼
        Concatenate([encoder_output_1, encoder_output_2])
                          ↓
               encoder_combined (24, 100)

────────────────────────────────────────────────────────────────────

   ┌────────────────────┐
   │ decoder_input (36) │
   └──────────┬─────────┘
              ▼
      ┌───────────────────────┐
      │ Embedding (vocab_a, 50)│
      └──────────┬────────────┘
                 ▼
         ┌──────────────┐         ┌──────────────┐
         │ Decoder LSTM1│         │ Decoder LSTM2│
         └─────┬────────┘         └─────┬────────┘
               │ decoder_output_1       │ decoder_output_2
               └────────┬───────────────┘
                        ▼
        Concatenate([decoder_output_1, decoder_output_2])
                          ↓
               decoder_combined (36, 100)

────────────────────────────────────────────────────────────────────

     decoder_combined ─┐
                        │
                        ▼
       AdditiveAttention([decoder_combined, encoder_combined])
                        ↓
               context_vector (36, 100)
                        ↓
               Dropout(0.2) → Dense(50) → 조정된 context_vector
                        ↓
   Concatenate([decoder_combined, context_vector])
                        ↓
               decoder_context_concat (36, 150)
                        ↓
     TimeDistributed(Dense(vocab_size)) → 최종 예측 (36, vocab_size)
