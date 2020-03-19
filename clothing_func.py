import numpy as np
from keras.utils import to_categorical
from keras.datasets import cifar10
from keras.optimizers import Adam

from keras.layers import Input, Flatten, Dense
from keras.models import Model


(xTrain, yTrain), (xTest, yTest) = cifar10.load_data()

NUM_CLASSES = len(set(yTrain.T[0]))

xTrain = xTrain / 255.0
xTest = xTest / 255.0

yTrain = to_categorical(yTrain, NUM_CLASSES)
yTest = to_categorical(yTest, NUM_CLASSES)

input_layer = Input(shape=(32, 32, 3))

x = Flatten()(input_layer)

x = Dense(units=200, activation='relu')(x)
x = Dense(units=150, activation='relu')(x)

output_layer = Dense(units=NUM_CLASSES, activation='softmax')(x)

model = Model(input_layer, output_layer)

optimizer = Adam(lr=0.0005)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.fit(xTrain, yTrain, batch_size=32, epochs=10, shuffle=True)

print(model.evaluate(xTest, yTest))

import pdb; pdb.set_trace()