import sys
import numpy as np
import scipy.special


class Sigmoid:
    def forward(self, x):
        self.outputs = scipy.special.expit(x)
        return self.outputs

    def backward(self, delta):
        return self.outputs * (1 - self.outputs) * delta


class Softmax:
    def forward(self, x):
        vector = np.exp(x - x.max(axis=-1, keepdims=True))
        return vector / vector.sum(axis=-1, keepdims=True)

    def backward(self, delta):
        return delta


def binary_loss(y_label, y_pred):
    batch_size = y_label.shape[0]
    return -np.sum(y_label * np.log(y_pred)) / batch_size


def accuracy_score(y_label, y_pred):
    return np.mean(y_label == y_pred)


train_image_file, train_label_file, test_image_file = sys.argv[1:]
train_image = np.genfromtxt(train_image_file, delimiter=',')
train_label = np.genfromtxt(train_label_file, delimiter=',')
test_image = np.genfromtxt(test_image_file, delimiter=',')

hidden_layer1_num = 256
hidden_layer2_num = 128
batch_size = 512
epoch_num = 300
learning_rate = 0.15
learning_rate_decay = 0.996
label_count = 10


class LinearLayer:
    def __init__(self, input_shape, activate_function):
        a, b = input_shape
        self.weight = np.random.randn(a, b) / np.sqrt(a)
        self.bias = np.zeros(b)
        self.activate_function = activate_function

    def forward(self, inputs):
        self.inputs = inputs
        return self.activate_function.forward(np.dot(inputs, self.weight) + self.bias)

    def backward(self, delta):
        delta = self.activate_function.backward(delta)
        batch_size = delta.shape[0]
        self.weight_grad = self.inputs.T.dot(delta) / batch_size
        self.bias_grad = np.sum(delta, axis=0) / batch_size
        return delta.dot(self.weight.T)

    def update(self, learning_rate):
        self.weight -= learning_rate * self.weight_grad
        self.bias -= learning_rate * self.bias_grad


train_data_size, train_input_size = train_image.shape

layers = []
layers.append(LinearLayer((train_input_size, hidden_layer1_num), Sigmoid()))
layers.append(LinearLayer((hidden_layer1_num, hidden_layer2_num), Sigmoid()))
layers.append(LinearLayer((hidden_layer2_num, label_count), Softmax()))

ohe_labels = np.eye(label_count)[train_label.astype(np.int32)]

for epoch in range(epoch_num):
    epoch_losses, epoch_accuracies = [], []
    # shuffle batch data
    random_permutation = np.random.permutation(train_data_size)
    for i in range(0, train_data_size, batch_size):
        batch_slice = random_permutation[slice(i, min(i + batch_size, train_data_size))]
        batch_X, batch_y = train_image[batch_slice], ohe_labels[batch_slice]

        # forward
        output = batch_X
        for layer in layers:
            output = layer.forward(output)

        # calcucate loss and accuracy
        epoch_losses.append(binary_loss(batch_y, output))
        epoch_accuracies.append(accuracy_score(train_label[batch_slice], np.argmax(output, axis=1)))

        # backward
        delta = output - batch_y
        for layer in reversed(layers):
            delta = layer.backward(delta)

        # update weights using gradient descent
        for layer in layers:
            layer.update(learning_rate)

    print('{}: loss: {}, acccuracy: {}'.format(epoch, np.mean(epoch_losses), np.mean(epoch_accuracies)))

    # decay the learning rate
    learning_rate *= learning_rate_decay


# predict on test images
output = test_image
for layer in layers:
    output = layer.forward(output)
pred_label = np.argmax(output, axis=1)

with open('test_predictions.csv', 'w') as f:
    for label in pred_label:
        f.write(str(label) + '\n')
