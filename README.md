# 모델 아키텍처

| 레이어 이름              | 출력 형상 (Shape)                | 레이어 종류           | 설명                                           |
|--------------------|----------------------------|------------------|----------------------------------------------|
| **Input (encoder_input)**  | (None, max_len_q)            | 입력 레이어         | 질문 시퀀스를 입력받는 레이어                           |
| **Embedding (encoder_emb)**| (None, max_len_q, 50)        | 임베딩 레이어        | 질문을 임베딩하여 고차원 벡터로 변환                       |
| **LSTM (encoder_lstm_1)**  | (None, max_len_q, 50)        | LSTM               | 질문을 처리하고, 은닉 상태 (`state_h`)와 셀 상태 (`state_c`) 반환      |
| **LSTM (decoder_lstm_1)**  | (None, max_len_a, 50)        | LSTM               | 디코더 첫 번째 LSTM, 인코더 상태를 초기 상태로 사용          |
| **LSTM (decoder_lstm_2)**  | (None, max_len_a, 50)        | LSTM               | 디코더 두 번째 LSTM, 첫 번째 LSTM 상태를 초기 상태로 사용   |
| **Concatenate (decoder_combined)** | (None, max_len_a, 100)      | 결합 레이어           | 두 개의 LSTM 출력을 결합하여 100 차원으로 만듦             |
| **Dense (encoder_output_1)** | (None, max_len_q, 100)      | 밀집 레이어 (TimeDistributed) | 인코더 출력에 Dense 레이어를 적용하여 100 차원으로 변환     |
| **Attention (attention_layer)** | (None, max_len_a, 100)      | 어텐션 레이어        | 디코더 출력과 인코더 출력을 기반으로 context vector 생성      |
| **Concatenate (decoder_context_concat)** | (None, max_len_a, 200)     | 결합 레이어           | 디코더 출력과 context vector를 결합                       |
| **Dense (decoder_dense)**   | (None, max_len_a, vocab_size_a) | 밀집 레이어 (TimeDistributed) | 각 타임스텝에서 vocab_size_a 크기의 출력 벡터를 생성하여 확률 분포 계산  |
| **Output (decoder_outputs)** | (None, max_len_a, vocab_size_a) | 출력 레이어         | 각 타임스텝에서 단어에 대한 softmax 확률 예측 결과            |

