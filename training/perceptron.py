import os
import cv2
import numpy as np

data = []
labels = []

classes = ["plastic", "paper", "metal"]

for label, folder in enumerate(classes):

    path = f"dataset/{folder}"

    for file in os.listdir(path):

        img_path = os.path.join(path, file)

        img = cv2.imread(img_path)

        img = cv2.resize(img, (64, 64))

        img = img.flatten() / 255.0

        data.append(img)

        labels.append(label)

X = np.array(data)
y = np.array(labels)

print(X.shape)
print(y.shape)

class Perceptron:

    def __init__(self, input_size):

        self.weights = np.random.randn(input_size)

        self.bias = np.random.randn()

    def step_function(self, x):

        return 1 if x > 0 else 0

    def predict(self, x):

        linear = np.dot(x, self.weights) + self.bias

        return self.step_function(linear)
    
model = Perceptron(X.shape[1])

for epoch in range(10):

    correct = 0

    for i in range(len(X)):

        prediction = model.predict(X[i])

        if prediction == y[i]:
            correct += 1

    accuracy = correct / len(X)

    print(f"Epoch {epoch+1} Accuracy: {accuracy}")