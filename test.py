# import cProfile
# from bs4 import BeautifulSoup
#
# def foo():
#     soup = BeautifulSoup('<div class="tbody m-tcol-c" id="tbody" style="width:744px; padding-left:43px; padding-right:43px; margin-right:0px;"> <p>ㅌㅅ 4800원으로 사봐요</p>  </div>', 'html.parser')
#     print(soup.get_text())
#     text2 = '세베 받는게 맞겠죠? <img src="https://cafeptthumb-phinf.pstatic.net/MjAxNzA2MDFfNTYg/MDAxNDk2MjQzNzk1NDA4.JJ3CN_VfIjDh6VmeI8tZFwWm-WzIxdswXzjjQyT3YAsg.p40ViB_PCTKVx6Q6fxWUhIkQaAi9NPq2ykDjH8neWhMg.JPEG.bsh1584/default.jpeg?type=w740" id="userImg3983563"style="width:740px; height:416px;"   name="cafeuserimg"  onclick="popview(this)">html테스트해볼까? &nbsp; fjf &nbsp;ds '
#     soup = BeautifulSoup(text2, 'html.parser')
#     print(soup.get_text())
# cProfile.run('foo()')




from __future__ import print_function
import numpy as np

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.datasets import imdb


max_features = 20000
# cut texts after this number of words
# (among top max_features most common words)
maxlen = 100
batch_size = 32

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
y_train = np.array(y_train)
y_test = np.array(y_test)

model = Sequential()
model.add(Embedding(max_features, 128, input_length=maxlen))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=4,
          validation_data=[x_test, y_test])