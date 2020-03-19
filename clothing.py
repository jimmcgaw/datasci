import tensorflow as tf
from tensorflow import keras


fashion_mnist = keras.datasets.fashion_mnist

(xTrain, yTrain), (xTest, yTest) = fashion_mnist.load_data()

# normalist the training data
xTrain = xTrain / 255.0
xTest = xTest / 255.0

model = keras.Sequential([
    # flatten the input from 2-d array to 1-d input vector
    keras.layers.Flatten(input_shape=(28, 28)),
    # first layer is fully connected with 128 nodes
    # activation is Rectified Linear Unit
    keras.layers.Dense(128, activation='relu'),
    # next layer has ten nodes, to correspond with number of classification categories
    keras.layers.Dense(10)
])

# Adam is algorithm for gradient-based optimization
model.compile(optimizer='adam',
              # loss func we'll minimize during training
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              # metric for which we're optimizing the loss function
              metrics=['accuracy'])

# train the model
model.fit(xTrain, yTrain, epochs=10)

test_loss, test_accuracy = model.evaluate(xTest, yTest, verbose=2)

print('Test loss: {}'.format(test_loss))
print('Test accuracy: {}'.format(test_accuracy))

import pdb; pdb.set_trace()

