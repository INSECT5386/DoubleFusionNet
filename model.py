import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, TimeDistributed, AdditiveAttention, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Attention
from tensorflow.keras import initializers
# 데이터 읽기
data = []
with open('/content/drive/MyDrive/DailoSeq_DATASET.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line))

questions = [item['question'] for item in data]
answers = [str(item['answer']) for item in data]

# 시작과 종료 토큰 추가
answers_input = ['<start> ' + answer for answer in answers]
answers_output = [answer + ' <end>' for answer in answers]

# 토크나이저 설정
tokenizer_q = Tokenizer()
tokenizer_q.fit_on_texts(questions)

tokenizer_a = Tokenizer()
tokenizer_a.fit_on_texts(answers_input + answers_output)

def save_tokenizer(tokenizer, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(tokenizer.to_json(), f, ensure_ascii=False)

save_tokenizer(tokenizer_q, '/content/drive/MyDrive/tokenizer_q.json')
save_tokenizer(tokenizer_a, '/content/drive/MyDrive/tokenizer_a.json')

# 시퀀스 변환 및 패딩
sequences_q = tokenizer_q.texts_to_sequences(questions)
sequences_a_input = tokenizer_a.texts_to_sequences(answers_input)
sequences_a_output = tokenizer_a.texts_to_sequences(answers_output)

max_len_q = max(len(seq) for seq in sequences_q)
max_len_a = max(len(seq) for seq in sequences_a_input)
print(max_len_q)
print(max_len_a)
X = pad_sequences(sequences_q, maxlen=max_len_q, padding='post')
y = pad_sequences(sequences_a_input, maxlen=max_len_a, padding='post')
y_output = pad_sequences(sequences_a_output, maxlen=max_len_a, padding='post')

# 데이터 타입 변환 (int32)
X = X.astype(np.int32)
y = y.astype(np.int32)
y_output = y_output.astype(np.int32)

vocab_size_q = len(tokenizer_q.word_index) + 1
vocab_size_a = len(tokenizer_a.word_index) + 1
from tensorflow.keras import initializers
from tensorflow.keras.layers import Input, Embedding, LSTM, Concatenate, Attention, TimeDistributed, Dense

# 인코더
encoder_input = Input(shape=(max_len_q,))
encoder_emb = Embedding(vocab_size_q, 50)(encoder_input)

encoder_lstm_1 = LSTM(50, return_sequences=True, return_state=True, name='encoder_lstm_1')
encoder_output_1, state_h, state_c = encoder_lstm_1(encoder_emb)

# 디코더
decoder_input = Input(shape=(max_len_a,))
decoder_emb = Embedding(vocab_size_a, 50)(decoder_input)

# 첫 번째 LSTM (초기 상태는 encoder에서 나오는 상태 사용)
decoder_lstm_1 = LSTM(50, return_sequences=True, return_state=True, name='decoder_lstm_1',
                      kernel_initializer=initializers.GlorotUniform(), recurrent_initializer=initializers.Orthogonal())
decoder_output_1, state_h_1, state_c_1 = decoder_lstm_1(decoder_emb, initial_state=[state_h, state_c])

# 두 번째 LSTM (다른 가중치 초기화)
decoder_lstm_2 = LSTM(50, return_sequences=True, return_state=True, name='decoder_lstm_2',
                      kernel_initializer=initializers.RandomNormal(), recurrent_initializer=initializers.Identity())
decoder_output_2, state_h_2, state_c_2 = decoder_lstm_2(decoder_emb, initial_state=[state_h_1, state_c_1])

# 두 LSTM의 출력을 결합하여 차원을 100으로 만듬
decoder_combined = Concatenate(axis=-1)([decoder_output_1, decoder_output_2])

# encoder_output_1에 Dense 레이어 적용 (각 타임스텝에 대해 100 차원으로 변경)
encoder_output_1 = TimeDistributed(Dense(100))(encoder_output_1)

# 어텐션 레이어 적용
attention_layer = Attention()
context_vector = attention_layer([decoder_combined, encoder_output_1])

# 디코더의 출력과 context vector를 결합
decoder_context_concat = Concatenate(axis=-1)([decoder_combined, context_vector])

# 출력층 (TimeDistributed를 사용해 각 타임스텝에 대해 Dense 레이어 적용)
decoder_dense = TimeDistributed(Dense(vocab_size_a, activation='softmax'))
decoder_outputs = decoder_dense(decoder_context_concat)

# 모델 정의
model = Model(inputs=[encoder_input, decoder_input], outputs=decoder_outputs)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 조기 종료 및 체크포인트 설정
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
checkpoint = ModelCheckpoint('/content/drive/MyDrive/model_checkpoint.keras', save_best_only=True)

# ✅ Eager Execution 활성화 (Graph Mode 문제 방지)
tf.config.run_functions_eagerly(True)

# ✅ 데이터셋 대신 Numpy 배열 사용
model.fit(
    [X, y], y_output,
    batch_size=20,
    epochs=1,
    validation_split=0.2,
    verbose=1,
    callbacks=[early_stopping, checkpoint]
)

# 모델 저장
model.save('/content/drive/MyDrive/2E-DcoderChat.keras')
