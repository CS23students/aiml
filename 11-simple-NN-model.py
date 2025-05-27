import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# Define the input and output data (XOR problem)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Define the model
model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=1000, batch_size=4, verbose=0)

# Evaluate the model on new data
test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
predictions = model.predict(test_data)

# Print predictions
print("Predictions:")
print(predictions)


# Predictions:
# [[0.01]
#  [0.98]
#  [0.98]
#  [0.02]]
