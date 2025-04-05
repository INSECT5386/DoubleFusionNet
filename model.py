import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, TimeDistributed, Attention, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import ZeroPadding1D
from tensorflow.keras.layers import AdditiveAttention
from tensorflow.keras.layers import Dense, Dropout
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

save_tokenizer(tokenizer_q, '/content/drive/MyDrive/TriFusionNet_tokenizer_q.json')
save_tokenizer(tokenizer_a, '/content/drive/MyDrive/TriFusionNet_tokenizer_a.json')

# 시퀀스 변환 및 패딩
sequences_q = tokenizer_q.texts_to_sequences(questions)
sequences_a_input = tokenizer_a.texts_to_sequences(answers_input)
sequences_a_output = tokenizer_a.texts_to_sequences(answers_output)

max_len_q = max(len(seq) for seq in sequences_q)
max_len_a = max(len(seq) for seq in sequences_a_input)

X = pad_sequences(sequences_q, maxlen=max_len_q, padding='post')
y = pad_sequences(sequences_a_input, maxlen=max_len_a, padding='post')
y_output = pad_sequences(sequences_a_output, maxlen=max_len_a, padding='post')

# 데이터 타입 변환 (int32)
X = X.astype(np.int32)
y = y.astype(np.int32)
y_output = y_output.astype(np.int32)

vocab_size_q = len(tokenizer_q.word_index) + 1
vocab_size_a = len(tokenizer_a.word_index) + 1

# 인코더
encoder_input = Input(shape=(max_len_q,))
encoder_emb = Embedding(vocab_size_q, 50)(encoder_input)

encoder_lstm_1 = LSTM(50, return_sequences=True, return_state=True, name='encoder_lstm_1')
encoder_output_1, state_h_1, state_c_1 = encoder_lstm_1(encoder_emb)

encoder_lstm_2 = LSTM(50, return_sequences=True, return_state=True, name='encoder_lstm_2')
encoder_output_2, state_h_2, state_c_2 = encoder_lstm_2(encoder_emb)

# 디코더
decoder_input = Input(shape=(max_len_a,))
decoder_emb = Embedding(vocab_size_a, 50)(decoder_input)

# 초기 상태를 encoder의 최종 상태로 전달
decoder_lstm_1 = LSTM(50, return_sequences=True, return_state=True, name='decoder_lstm_1')
decoder_output_1, _, _ = decoder_lstm_1(decoder_emb, initial_state=[state_h_1, state_c_1])

decoder_lstm_2 = LSTM(50, return_sequences=True, return_state=True, name='decoder_lstm_2')
decoder_output_2, _, _ = decoder_lstm_2(decoder_emb, initial_state=[state_h_2, state_c_2])

# 인코더 & 디코더 출력 결합
encoder_combined = Concatenate(axis=-1)([encoder_output_1, encoder_output_2])
decoder_combined = Concatenate(axis=-1)([decoder_output_1, decoder_output_2])


context_vector = AdditiveAttention()([decoder_combined, encoder_combined])
context_vector = Dropout(0.2)(context_vector)
context_vector = Dense(50, activation='tanh')(context_vector)  # 정보 조정

# 최종 결합
decoder_context_concat = Concatenate(axis=-1)([decoder_combined, context_vector])


# 출력층
decoder_dense = TimeDistributed(Dense(vocab_size_a, activation='softmax'))
decoder_outputs = decoder_dense(decoder_context_concat)

# 모델 정의
model = Model([encoder_input, decoder_input], decoder_outputs)

# 모델 컴파일
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 모델 요약
model.summary()

# 모델 학습
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
model_checkpoint = ModelCheckpoint('/content/drive/MyDrive/TriFusionNet_best_model.h5', save_best_only=True)

history = model.fit([X, y], np.expand_dims(y_output, -1), epochs=1, batch_size=16, validation_split=0.2, callbacks=[early_stopping, model_checkpoint])

model.save('/content/drive/MyDrive/TriFusionNet_model.h5')
