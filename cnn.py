import numpy as np
import mnist
from conv import Conv3x3
from maxpool import Maxpool2x2
from softmax import Softmax
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', as_frame=False)

X, y = mnist.data, mnist.target.astype(int)

np.random.seed(42)

conv = Conv3x3(8)
pool = Maxpool2x2()
softmax = Softmax(13*13*8, 10)

def forward(image, label):
    image = image.reshape(28, 28)
    output = conv.forward((image / 255) - 0.5)
    output = pool.forward(output)
    output = softmax.forward(output)

    loss = -np.log(output[label])
    acc = 1 if np.argmax(output) == label else 0
    return output, loss, acc

def train(image, label, lr=0.01):
    out, loss, acc = forward(image, label)
    gradient = np.zeros(10)
    gradient[label] = -1/out[label]
    gradient = softmax.backprop(gradient, lr)
    gradient = pool.backprop(gradient)
    gradient = conv.backprop(gradient, lr)
    return loss, acc

print('MNIST CNN initialized!')

for epoch in range(3):
    print(f'Epoch {epoch + 1}')

    loss = 0
    num_correct = 0
    for i, (im, label) in enumerate(zip(X[:15000], y[:15000])):
        if i % 100 == 99:
            print(f'[Step {i + 1}] Past 100 steps: Average Loss {loss / 100:.3f} | Accuracy: {num_correct}%')
            loss = 0
            num_correct = 0
        l, acc = train(im, label)
        loss += l
        num_correct += acc
print("TEST CNN")
loss = 0
num_correct = 0
for im, label in zip(X[60000:], y[60000:]):
    _, l, acc = forward(im, label)
    loss += l
    num_correct += acc
num_tests = len(X[60000:])
print('Test Loss:', loss / num_tests)
print('Test Accuracy:', num_correct / num_tests)



