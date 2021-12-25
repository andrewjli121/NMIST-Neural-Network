import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#Largely inspired by
#https://www.youtube.com/watch?v=w8yWXqWQYmU

data = pd.read_csv('train.csv')

data = np.array(data)

m, n = data.shape
np.random.shuffle(data)

data = data.T

#x_data = data[1:, 1:2] / 255
#y_data = data[0, 1:2]

x_data = data[1:] / 255
y_data = data[0]

def init():
    weightsOne = np.random.rand(10,784) - 0.5
    weightsTwo = np.random.rand(10,10) - 0.5
    biasOne = np.random.rand(10,1) - 0.5
    biasTwo = np.random.rand(10,1) - 0.5
    return weightsOne, biasOne, weightsTwo, biasTwo

def relu(x):
    return np.maximum(x,0)

def relu_deriv(x):
    return x > 0

def softmax(x):
    a = np.exp(x) / sum(np.exp(x))
    return a

def forward_prop(weightsOne, biasOne, weightsTwo, biasTwo, x):
    W1 = weightsOne.dot(x) + biasOne
    L1 = relu(W1)
    W2 = weightsTwo.dot(L1) + biasTwo
    L2 = softmax(W2)
    return W1, L1, W2, L2

def oneZero(x):
    one = np.zeros((x.size, 10))
    one[np.arange(x.size), x] = 1
    one = one.T
    return one

def back_prop(W1, L1, W2, L2, weightsOne, weightsTwo, x, y):
    supposed = oneZero(y)
    dW2 = L2 - supposed
    dweightsTwo = 1 / m * dW2.dot(L1.T)
    dbiasTwo = 1 / m * np.sum(dW2)
    dW1 = weightsTwo.T.dot(dW2) * relu_deriv(W1)
    dweightsOne = 1 / m * dW1.dot(x.T)
    dbiasOne = 1 / m * np.sum(dW1)
    return dweightsOne, dbiasOne, dweightsTwo, dbiasTwo

def update(weightsOne, biasOne, weightsTwo, biasTwo, dweightsOne, dbiasOne, dweightsTwo, dbiasTwo, alpha):
    weightsOne = weightsOne - alpha * dweightsOne
    biasOne = biasOne - alpha * dbiasOne
    weightsTwo = weightsTwo - alpha * dweightsTwo
    biasTwo = biasTwo - alpha * dbiasTwo
    return weightsOne, biasOne, weightsTwo, biasTwo

def neural_net(x, y, alpha, iterations):
    weightsOne, biasOne, weightsTwo, biasTwo = init()
    for i in range(iterations):
        W1, L1, W2, L2 = forward_prop(weightsOne, biasOne, weightsTwo, biasTwo, x)
        dweightsOne, dbiasOne, dweightsTwo, dbiasTwo = back_prop(W1, L1, W2, L2, weightsOne, weightsTwo, x, y)
        weightsOne, biasOne, weightsTwo, biasTwo = update(weightsOne, biasOne, weightsTwo, biasTwo, dweightsOne, dbiasOne, dweightsTwo, dbiasTwo, alpha)
        if(i % 100 == 0):
            print('Error: ', np.mean(np.abs(L2 - oneZero(y))))
    return weightsOne, biasOne, weightsTwo, biasTwo

W1, b1, W2, b2 = neural_net(x_data, y_data, 0.25, 500)

test_data = pd.read_csv('test.csv')
test_data = np.array(test_data)
test_data = test_data.T

test_x = test_data[0:]
test_y = test_data[0]

def test(num, W1, b1, W2, b2, data):
    image = data[:, num, None].reshape((28,28)) * 255

    _, _, _, x = forward_prop(W1, b1, W2, b2, data[:, num, None])

    plt.gray()
    plt.imshow(image, interpolation= 'nearest')
    plt.show()
    print(np.argmax(x,0))
    return np.argmax(x, 0)
    
test(4, W1, b1, W2, b2, test_x)
test(6, W1, b1, W2, b2, test_x)
test(7, W1, b1, W2, b2, test_x)
test(8, W1, b1, W2, b2, test_x)
test(9, W1, b1, W2, b2, test_x)
test(14, W1, b1, W2, b2, test_x)
