# INSTALL this :  pip install numpy tensorflow keras

import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# Define the input and output data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])  # XOR output

# Define the model
model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))  # Hidden layer with 4 neurons
model.add(Dense(1, activation='sigmoid'))            # Output layer

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model to the data
model.fit(X, y, epochs=1000, batch_size=4, verbose=1)

# Evaluate the model
loss, accuracy = model.evaluate(X, y, verbose=0)
print(f"\nFinal Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

# Make predictions
test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
predictions = model.predict(test_data)

print("\nPredictions:")
for i, pred in enumerate(predictions):
    print(f"Input: {test_data[i]} -> Predicted: {pred[0]:.4f}")


# 3. Save and Run the Program
# Open any text editor or IDE (like VS Code, PyCharm, or Notepad++).

# Paste the code into a new file and save it as xor_nn.py.
# Then open your terminal or command prompt, go to the directory where you saved the file, and run:
# python xor_nn.py


# OP:
# Epoch 1000/1000
# 1/1 [==============================] - 0s 4ms/step - loss: 0.0132 - accuracy: 1.0000

# Final Loss: 0.0132, Accuracy: 1.0000

# 1/1 [==============================] - 0s 57ms/step

# Predictions:
# Input: [0 0] -> Predicted: 0.0152
# Input: [0 1] -> Predicted: 0.9798
# Input: [1 0] -> Predicted: 0.9804
# Input: [1 1] -> Predicted: 0.0219
