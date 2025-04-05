  ┌────────────┐         ┌────────────────────┐         ┌────────────┐
  │ encoder_input (24) ─▶│  Embedding (50)     ├────┐    │            │
  └────────────┘         └────────────────────┘    │    ▼            ▼
                                                 ┌───────┐      ┌────────┐
                                                 │ LSTM1 │      │ LSTM2  │
                                                 └──┬────┘      └──┬─────┘
                                                    │              │
             ┌────────────────────┐               encoder_output_1 │
             │ decoder_input (36) ├───▶ Embedding ────────────────┼───▶ Concatenate → encoder_combined (24, 100)
             └────────────────────┘              (50 dim)        encoder_output_2 │
                                                       ▼                          ▼

                                                   ┌────────┐      ┌────────┐
                                                   │LSTM1   │      │LSTM2   │
                                        state_h/c◀─┴        │      │        ├──▶ decoder_output_2
                                      (전달됨)              └────────┘      └────────┘

                                                     decoder_output_1 ─────┐
                                                     decoder_output_2 ─────┘ → decoder_combined (36, 100)

    decoder_combined → AdditiveAttention([dec, enc]) → context_vector (36, 100)
                      → Dropout(0.2) → Dense(50) → 조정된 컨텍스트

    Concatenate([decoder_combined, context_vector]) → (36, 150)
    → TimeDistributed(Dense(vocab_size)) → 최종 예측 (36, vocab_size) 이렇게?
