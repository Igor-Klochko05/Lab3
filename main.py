import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical
from keras.optimizers import Adam


# Download training and test data
train_data = pd.read_csv('mnist_train.csv')
test_data = pd.read_csv('mnist_test.csv')

# Separation into data and labels
X_train = train_data.iloc[:, 1:].values.reshape(-1, 28, 28, 1) / 255.0
y_train = to_categorical(train_data.iloc[:, 0].values, num_classes=10)

X_test = test_data.iloc[:, 1:].values.reshape(-1, 28, 28, 1) / 255.0
y_test = to_categorical(test_data.iloc[:, 0].values, num_classes=10)

# Creating a model
model = Sequential()
model.add(Flatten(input_shape=(28, 28, 1)))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Training the model
model.fit(X_train, y_train, epochs=10, batch_size=200, validation_data=(X_test, y_test))

# Predictions based on test data
predictions = model.predict(X_test)

# Output of results and display of numbers
for i in range(len(predictions)):
    true_label = np.argmax(y_test[i])
    predicted_label = np.argmax(predictions[i])

    if true_label == predicted_label:
        print(f"Value {true_label} was correctly predicted as {predicted_label}")
    else:
        print(f"Value {true_label} was incorrectly predicted as {predicted_label}")

    # Display a number
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"True: {true_label}, Predicted: {predicted_label}")
    plt.pause(1.1)  # Додає затримку для того, щоб зображення відобразилося перед продовженням виконання коду
    plt.close()
