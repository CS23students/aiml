import numpy as np
from keras.models import Sequential
from keras.layers import Dense
Define the input and output data X
= np.array([[0, 0], [0, 1], [1, 0], [1,
1]])
y = np.array([[0], [1], [1], [0]])
Define the model
model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
EX NO.: 11

Build Simple NN Models

DATE :

Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
Fit the model to the data
model.fit(X, y, epochs=1000, batch_size=4)
Evaluate the model on new data test_data =
np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
predictions = model.predict(test_data)
print(predictions)

# OUTPUT: 1/1 [==============================] - ETA: 0s - loss: 0.7197 - accuracy: 0.7500
# 1/1 [==============================] - 0s 417ms/step - loss: 0.7197 - accuracy: 0.7500
# 1/1 [==============================] - ETA: 0s
# 1/1 [==============================] - 0s 101ms/step
# [[0.5005405 ]
# [0.28815603]
# [0.6136732 ]
# [0.36250085]]
