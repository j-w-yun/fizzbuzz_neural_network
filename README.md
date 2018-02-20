# fizzbuzz_neural_network
## Approximating FizzBuzz
I am approximating the infamous FizzBuzz function:

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

From 1 to 100, the correct output should be:

    ['1' '2' 'Fizz' '4' 'Buzz' 'Fizz' '7' '8' 'Fizz' 'Buzz' '11' 'Fizz' '13'
     '14' 'FizzBuzz' '16' '17' 'Fizz' '19' 'Buzz' 'Fizz' '22' '23' 'Fizz'
     'Buzz' '26' 'Fizz' '28' '29' 'FizzBuzz' '31' '32' 'Fizz' '34' 'Buzz'
     'Fizz' '37' '38' 'Fizz' 'Buzz' '41' 'Fizz' '43' '44' 'FizzBuzz' '46' '47'
     'Fizz' '49' 'Buzz' 'Fizz' '52' '53' 'Fizz' 'Buzz' '56' 'Fizz' '58' '59'
     'FizzBuzz' '61' '62' 'Fizz' '64' 'Buzz' 'Fizz' '67' '68' 'Fizz' 'Buzz'
     '71' 'Fizz' '73' '74' 'FizzBuzz' '76' '77' 'Fizz' '79' 'Buzz' 'Fizz' '82'
     '83' 'Fizz' 'Buzz' '86' 'Fizz' '88' '89' 'FizzBuzz' '91' '92' 'Fizz' '94'
     'Buzz' 'Fizz' '97' '98' 'Fizz' 'Buzz']

My neural network is classifying each number into one of four categories:

    0. "Fizz"
    1. "Buzz"
    2. "FizzBuzz"
    3. None of the above

## Preparing Data
I am encoding the X (input) values as 16-bit binary:

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

And encoding the Y (output) values as one-hot vectors:

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

which will categorize the 16-bit binary input data as one of the 4 possible categories specified by the FizzBuzz rule.

For example, if `[ 0.03 -0.4 -0.4  0.4]` is returned, the program knows not to print any of "Fizz", "Buzz", or "FizzBuzz":

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

## Initializing Data
This is how I am dividing up the training and testing data:

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

so the model trains using values between 101 and 10000 and tests using values between 1 and 100.

## Neural Network Model
My model architecture is simple, with 100 hidden neurons in one layer:

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
    train_step = tf.train.AdamOptimizer(0.005).minimize(cross_entropy)
    
    # define accuracy
    correct_prediction = tf.equal(tf.argmax(Z, 1), tf.argmax(Y, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

## Running the Model
For the sake of simplicity, I opted to omit batch training:

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
    
        for i in range(10000):
            sess.run(train_step, feed_dict={X: train_x, Y: train_y})
    
            train_accuracy = sess.run(accuracy, feed_dict={X: train_x, Y: train_y})
            print(i, ":", train_accuracy)
    
        output = sess.run(Z, feed_dict={X: test_x})
        decoded = one_hot_decode_array(test_x_raw, output)
        print(decoded)

## Results
After about 5,000 iterations of train step, training accuracy converges to 1.0. Here is the following test output of the trained neural network model after 10,000 iterations:

        0 : 0.346061
        1 : 0.459596
        2 : 0.48404
        3 : 0.472828
        4 : 0.441515
        5 : 0.417071
        
        ...
        
        9998 : 1.0
        9999 : 1.0
        ['1' '2' 'Fizz' '4' 'Buzz' 'Fizz' '7' '8' 'Fizz' 'Buzz' '11' 'Fizz' '13'
         '14' 'FizzBuzz' '16' '17' 'Fizz' '19' 'Buzz' 'Fizz' '22' '23' 'Fizz'
         'Buzz' '26' 'Fizz' '28' '29' 'FizzBuzz' '31' '32' 'Fizz' '34' 'Buzz'
         'Fizz' '37' '38' 'Fizz' 'Buzz' '41' 'Fizz' '43' '44' 'FizzBuzz' '46' '47'
         'Fizz' '49' 'Buzz' 'Fizz' '52' '53' 'Fizz' 'Buzz' '56' 'Fizz' '58' '59'
         'FizzBuzz' '61' '62' 'Fizz' '64' 'Buzz' 'Fizz' '67' '68' 'Fizz' 'Buzz'
         '71' 'Fizz' '73' '74' 'FizzBuzz' '76' '77' 'Fizz' '79' 'Buzz' 'Fizz' '82'
         '83' 'Fizz' 'Buzz' '86' 'Fizz' '88' '89' 'FizzBuzz' '91' '92' 'Fizz' '94'
         'Buzz' 'Fizz' '97' '98' 'Fizz' 'Buzz']

This model, despite its simplicity, without any prior information about the modulo operation, successfully extrapolated the output of the FizzBuzz function in the domain that was excluded in its training data.
