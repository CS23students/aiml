# Program:
# Import necessary libraries
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
# Define the neural network model
model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10, activation='softmax'))
# Compile the model


model.compile(loss='categorical_crossentropy',
optimizer='sgd',
metrics=['accuracy'])
Generate some random data for training and testing data
= np.random.random((1000, 100))
labels = np.random.randint(10, size=(1000, 1))
one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)
Train the model on the data
model.fit(data, one_hot_labels, epochs=10, batch_size=32)
Evaluate the model on a test set test_data =
np.random.random((100, 100))
test_labels = np.random.randint(10, size=(100, 1))
test_one_hot_labels = keras.utils.to_categorical(test_labels, num_classes=10) loss_and_metrics =
model.evaluate(test_data, test_one_hot_labels, batch_size=32) print("Test loss:",
loss_and_metrics[0])
print("Test accuracy:", loss_and_metrics[1])
