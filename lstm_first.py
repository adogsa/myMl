import numpy as np
import keras
from keras.utils import np_utils
from keras.preprocessing import sequence

acc_data = np.load('/nfs/data/npy/cs/acc.npy')
bill_data = np.load('/nfs/data/npy/cs/bill.npy')
x_train = np.concatenate((acc_data[:20000],bill_data[:20000]), axis=0)
y_train = np.zeros(shape=(40000,), dtype=np.float32)
y_train[20000:] = 1

x_val = np.concatenate((acc_data[20000:23000],bill_data[20000:23000]), axis = 0)
y_val = np.zeros(shape=(6000,), dtype=np.float32)
y_val[3000:] = 1

x_test = np.concatenate((acc_data[23000:24000],bill_data[23000:24000]), axis=0)
y_test = np.zeros(shape=(2000,), dtype=np.float32)
y_test[1000:] = 1

from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_train)
x_train = tokenizer.texts_to_sequences(x_train)
x_val = tokenizer.texts_to_sequences(x_val)
x_test = tokenizer.texts_to_sequences(x_test)

x_train = sequence.pad_sequences(x_train, maxlen=200)
x_val = sequence.pad_sequences(x_val, maxlen=200)
x_test = sequence.pad_sequences(x_test, maxlen=200)


# one-hot 인코딩
y_train = np_utils.to_categorical(y_train)
y_val = np_utils.to_categorical(y_val)
y_test = np_utils.to_categorical(y_test)


from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import LSTM

model = Sequential()
model.add(Embedding(15000, 128, input_length=200))
# model.add(LSTM(1024))
model.add(LSTM(64))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
# 4. 모델 학습시키기
hist = model.fit(x_train, y_train, epochs=30, batch_size=256, validation_data=(x_val, y_val))