import numpy as np
import tensorflow as tf


def fizzbuzz(start, end):
    a = list()
    for i in range(start, end + 1):
        a.append(fb(i))
    return a


def fb(i):
    if i % 3 == 0 and i % 5 == 0:
        return "FizzBuzz"
    elif i % 3 == 0:
        return "Fizz"
    elif i % 5 == 0:
        return "Buzz"
    else:
        return i


# encoding values of X
def binary_encode_16b_array(a):
    encoded_a = list()
    for elem in a:
        encoded_a.append(binary_encode_16b(elem))
    return np.array(encoded_a)


def binary_encode_16b(val):
    bin_arr = list()
    bin_str = format(val, '016b')
    for bit in bin_str:
        bin_arr.append(bit)
    return np.array(bin_arr)


# encoding values of Y
def one_hot_encode_array(a):
    encoded_a = list()
    for elem in a:
        encoded_a.append(one_hot_encode(elem))
    return np.array(encoded_a)


def one_hot_encode(val):
    if val == 'Fizz':
        return np.array([1, 0, 0, 0])
    elif val == 'Buzz':
        return np.array([0, 1, 0, 0])
    elif val == 'FizzBuzz':
        return np.array([0, 0, 1, 0])
    else:
        return np.array([0, 0, 0, 1])


# decoding values of Y
def one_hot_decode_array(x, y):
    decoded_a = list()
    for index, elem in enumerate(y):
        decoded_a.append(one_hot_decode(x[index], elem))
    return np.array(decoded_a)


def one_hot_decode(x, val):
    index = np.argmax(val)
    if index == 0:
        return 'Fizz'
    elif index == 1:
        return 'Buzz'
    elif index == 2:
        return 'FizzBuzz'
    elif index == 3:
        return x


# train with data that will not be tested
test_x_start = 1
test_x_end = 100
train_x_start = 101
train_x_end = 10000

test_x_raw = np.arange(test_x_start, test_x_end + 1)
test_x = binary_encode_16b_array(test_x_raw).reshape([-1, 16])
test_y_raw = fizzbuzz(test_x_start, test_x_end)
test_y = one_hot_encode_array(test_y_raw)

train_x_raw = np.arange(train_x_start, train_x_end + 1)
train_x = binary_encode_16b_array(train_x_raw).reshape([-1, 16])
train_y_raw = fizzbuzz(train_x_start, train_x_end)
train_y = one_hot_encode_array(train_y_raw)

# define params
input_dim = 16
output_dim = 4
h1_dim = 100

# build graph
X = tf.placeholder(tf.float32, [None, input_dim])
Y = tf.placeholder(tf.float32, [None, output_dim])

h1_w = tf.Variable(tf.random_normal([input_dim, h1_dim], stddev=0.1))
h1_b = tf.Variable(tf.zeros([h1_dim]))
h1_z = tf.nn.relu(tf.matmul(X, h1_w) + h1_b)

fc_w = tf.Variable(tf.random_normal([h1_dim, output_dim], stddev=0.1))
fc_b = tf.Variable(tf.zeros([output_dim]))
Z = tf.matmul(h1_z, fc_w) + fc_b

# define cost
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=Z))

# define op
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# define accuracy
correct_prediction = tf.equal(tf.argmax(Z, 1), tf.argmax(Y, 1))
correct_prediction = tf.cast(correct_prediction, tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(10000):
        sess.run(train_step, feed_dict={X: train_x, Y: train_y})

        train_accuracy = sess.run(accuracy, feed_dict={X: train_x, Y: train_y})
        print(i, ":", train_accuracy)

    output = sess.run(Z, feed_dict={X: test_x})
    decoded = one_hot_decode_array(test_x_raw, output)
    print(decoded)

