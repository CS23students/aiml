# Program:

# Import necessary libraries
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

# Define the neural network model
model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=100))  # Input layer with 100 features
model.add(Dense(units=10, activation='softmax'))               # Output layer for 10 classes

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

# Generate random data for training
data = np.random.random((1000, 100))
labels = np.random.randint(10, size=(1000, 1))
one_hot_labels = to_categorical(labels, num_classes=10)

# Train the model
model.fit(data, one_hot_labels, epochs=10, batch_size=32)

# Generate random test data
test_data = np.random.random((100, 100))
test_labels = np.random.randint(10, size=(100, 1))
test_one_hot_labels = to_categorical(test_labels, num_classes=10)

# Evaluate the model
loss_and_metrics = model.evaluate(test_data, test_one_hot_labels, batch_size=32)
print("Test loss:", loss_and_metrics[0])
print("Test accuracy:", loss_and_metrics[1])



# Epoch 10/10
# 32/32 [==============================] - 0s 1ms/step - loss: 2.3021 - accuracy: 0.1040
# 4/4 [==============================] - 0s 2ms/step - loss: 2.3017 - accuracy: 0.1200
# Test loss: 2.3017
# Test accuracy: 0.12
