"""
참고 : https://github.com/philipperemy/keras-attention-mechanism
"""



import numpy as np
from keras import backend as K
from keras import models
from keras import layers
import matplotlib.pyplot as plt
import pandas as pd

data_count = 10000
input_dims = 32
attention_column = 7


def make_data(batch_size, input_size, attention_index):
    """
    학습 데이터를 만듭니다.
    한 배치만 보고 설명하자면 입력 데이터는 input_size 길이이며
    attention_index를 제외한 곳은 전부 랜덤한 수로 설정됩니다.
    목표 데이터는 0 또는 1이며 이 값과 입력 데이터의 attention_index 위치의 값은 같습니다.
    """
    train_x = np.random.standard_normal(size=(batch_size, input_size))
    train_y = np.random.randint(low=0, high=2, size=(batch_size, 1))
    train_x[:, attention_index] = train_y[:, 0]
    return (train_x, train_y)


# Input Layer
input_layer = layers.Input(shape=(input_dims,))

# Attention Layer
attention_probs = layers.Dense(input_dims, activation='softmax')(input_layer)
attention_mul = layers.multiply([input_layer, attention_probs])

# FC Layer
y = layers.Dense(64)(attention_mul)
y = layers.Dense(1, activation='sigmoid')(y)

model = models.Model(input_layer, y)

model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# Train
train_x, train_y = make_data(data_count, input_dims, attention_column)

model.fit(train_x, train_y, epochs=20, batch_size=64, validation_split=0.2,
          verbose=2)

# Test
test_x, test_y = make_data(data_count, input_dims, attention_column)

result = model.evaluate(test_x, test_y, batch_size=64, verbose=0)
print("Loss:", result[0])
print("Accuracy:", result[1])

# Get attention vector
attention_layer = model.layers[2]
func = K.function([model.input] + [K.learning_phase()], [attention_layer.output])
output = func([test_x, 1.0])[0]
attention_vector = np.mean(output, axis=0)

# Show attention vector
pd.DataFrame(attention_vector, columns=['attention (%)']).plot(kind='bar',
                                                               title='Attention Vector')
plt.show()