#!/usr/bin/python

import tensorflow as tf
import numpy as np
import sys

# iteratively read the data becasue my computer is old and can't take large data
def iter_loadtxt(filename, delimiter=',', skiprows=0, dtype=float):
    def iter_func():
        with open(filename, 'r') as infile:
            for _ in range(skiprows):
                next(infile)
            for line in infile:
                line = line.rstrip().split(delimiter)
                for item in line:
                    yield dtype(item)
        iter_loadtxt.rowlength = len(line)

    data = np.fromiter(iter_func(), dtype=dtype)
    data = data.reshape((-1, iter_loadtxt.rowlength))
    return data

# array of classes
d = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 24, 25, 27, 28, 30, 32, 35, 36, 40, 42, 45, 48, 49, 54, 56, 63, 64, 72, 81] 


# raw input to data that ranges 0 to 39
def raw_2_data(lst):
    for i in range(len(lst)):
        lst[i] = d.index(lst[i])

# range from 0 to 39 to real output
def data_2_raw(lst):
    for i in range(len(lst)):
        lst[i] = d[lst[i]]

# takes a sparse list, returns one-hot list
def one_hot(lst,num_classes):
    ret = []
    for elem in lst:
        v = [0]*num_classes
        v[int(elem)] = 1
        ret.append(v)
    return ret

# returns a batch given number, data and label
def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels. 
    '''
    # arange the index
    idx = np.arange(0 , len(data))

    # shuffle the index
    np.random.shuffle(idx)

    # cut the index to the size of the batch
    idx = idx[:num]

    # select data according to the index
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    # return the data and label
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

# get data from csvs
def get_data(dir_path = './data'):
    x_train = iter_loadtxt(dir_path+'/train_x.csv').reshape(-1, 64, 64, 1)
    x_test = iter_loadtxt(dir_path+'/test_x.csv').reshape(-1, 64, 64, 1)
    y_train = iter_loadtxt(dir_path+'/train_y.csv').reshape(-1)
    raw_2_data(y_train)
    y_train = one_hot(y_train,40)
    return x_train,y_train,x_test


# get data
x_train, y_train, x_test = get_data()


# Data preprocessing

# filter out by threshhold
threshhold = 230.0
x_test[x_test<threshhold] = 0.0
x_test[x_test>=threshhold] = 255.0
x_train[x_train<threshhold] = 0.0
x_train[x_train>= threshhold] = 255.0

# Normalization
x_train /= np.max(x_train)
x_test /= np.max(x_test)

# training validation split
x_val = x_train[45000:]
y_val = y_train[45000:]

x_train = x_train[:45000]
y_train = y_train[:45000]


# returns variable
def weight_variable(shape):

    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# returns bias
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# returns convlutional layer
def conv2d(x, W):
    return tf.nn.conv2d(x,
                        W,
                        strides=[1, 1, 1, 1],
                        padding='SAME')

# returns pooling layer
def max_pool_2x2(x):

    return tf.nn.max_pool(x,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')

def main():
    # session
    sess = tf.InteractiveSession()

    # placeholders; input, label rates for dropout layers
    x = tf.placeholder("float", shape=[None, 64, 64, 1])
    y_ = tf.placeholder("float", shape=[None, 40])
    keep_prob1 = tf.placeholder("float")
    keep_prob2 = tf.placeholder("float")

    # Convolutional Layer 1
    W_conv1 = weight_variable([3, 3, 1, 16])
    b_conv1 = bias_variable([16])
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    h_conv1_drop = tf.nn.dropout(h_pool1, keep_prob1)

    # Convolutional Layer 2
    W_conv2 = weight_variable([3, 3, 16, 32])
    b_conv2 = bias_variable([32])
    h_conv2 = tf.nn.relu(conv2d(h_conv1_drop, W_conv2) + b_conv2)

    # Convolutional Layer 3
    W_conv3 = weight_variable([3, 3, 32, 32])
    b_conv3 = bias_variable([32])
    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)

    # Convolutional Layer 4
    W_conv4 = weight_variable([3, 3, 32, 64])
    b_conv4 = bias_variable([64])
    h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)

    # Convolutional Layer 5
    W_conv5 = weight_variable([3, 3, 64, 128])
    b_conv5 = bias_variable([128])
    h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5) + b_conv5)
    h_pool5 = max_pool_2x2(h_conv5)

    # Fully connected layer 1
    W_fc1 = weight_variable([8*8*128, 1024])
    b_fc1 = bias_variable([1024])
    h_conv5_drop_flat = tf.reshape(h_pool5, [-1, 8*8*128])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv5_drop_flat, W_fc1) + b_fc1)

    # Dropout
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob2)

    # Fully connected layer 2
    W_fc2 = weight_variable([1024, 40])
    b_fc2 = bias_variable([40])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # cross entropy loss
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
    cross_entropy = tf.reduce_mean(cross_entropy)

    # we use ADAM with default settings
    train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

    # initialize all variables
    sess.run(tf.initialize_all_variables())

    # 750k iterations
    for i in range(750000):

        # get the batch
        batch = next_batch(100,x_train,y_train)

        # output the training accuracy every 1000 iterations
        if i%1000 == 0:
            train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob1: 1.0, keep_prob2: 1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy))
        
        # train the model
        sess.run(train_step, {x: batch[0], y_: batch[1], keep_prob1: 0.25, keep_prob2: 0.5})

    # get the validation accuracy
    val_accuracy = accuracy.eval(feed_dict={x:x_val, y_: y_val, keep_prob1: 1.0, keep_prob2: 1.0})
    print("validation accuracy {0}".format(val_accuracy))

    # separate the test data because my computer cannot take them at once
    fin1 = sess.run(tf.argmax(y_conv,1), {x: x_test[0:2500], keep_prob1: 1.0, keep_prob2: 1.0}).tolist()
    fin2 = sess.run(tf.argmax(y_conv,1), {x: x_test[2500:5000], keep_prob1: 1.0, keep_prob2: 1.0}).tolist()
    fin3 = sess.run(tf.argmax(y_conv,1), {x: x_test[5000:7500], keep_prob1: 1.0, keep_prob2: 1.0}).tolist()
    fin4 = sess.run(tf.argmax(y_conv,1), {x: x_test[7500:10000], keep_prob1: 1.0, keep_prob2: 1.0}).tolist()

    # concatinate the test result
    fin = fin1 + fin2 + fin3 + fin4

    # change the index to the output and print them
    data_2_raw(fin)
    with open("./out.csv", "w+") as f:
        for e in fin:
            f.write(str(e)+'\n')

if __name__ == "__main__":
    main()