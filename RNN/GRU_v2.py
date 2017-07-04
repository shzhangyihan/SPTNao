import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import csv
import cv2

np.set_printoptions(threshold = 'nan')

# hyperparameters
learning_rate = 0.001
num_epochs = 150
num_epochs_other = 300
total_length = 840
input_size = 4 + 40
output_size = input_size
cell_size = 4 * input_size
num_layer = 4
num_batch = 2
batch_size = 1
seq_len = 70

root_path = '../Nao_data/resize_4bit_noisy_40/'
joint_path = '../Nao_data/nao_data_joint.csv'

# GRU graph input
x_placeholder = tf.placeholder(tf.float32, [None, seq_len, input_size])
y_placeholder = tf.placeholder(tf.float32, [None, seq_len, output_size])
Y_ = tf.reshape(y_placeholder, [-1, output_size])
Hin = tf.placeholder(tf.float32, [None, cell_size * num_layer])

# GRU model
mcell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(cell_size) for _ in range(num_layer)], state_is_tuple = False)
out, state = tf.nn.dynamic_rnn(mcell, x_placeholder, initial_state = Hin)

# output
Y = tf.contrib.layers.linear(tf.reshape(out, [-1, cell_size]), output_size)

# loss and optimizer
loss = tf.contrib.losses.mean_squared_error(Y, Y_)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

def readData():
    x = np.zeros([num_batch, batch_size, total_length, input_size])
    y = np.zeros([num_batch, batch_size, total_length, input_size])
    actual = np.zeros([num_batch, total_length, input_size])
    for batch in range(num_batch):
        b = (batch - 1) % 9
        file_path = root_path + 'encode_' + str(b) + '.txt'
        # is self, load joint
        if b == 8:
            array = []
            f = open(joint_path, 'rb')
            dataReader = csv.reader(f)
            for row in dataReader:
                array.append(row)
            data = map(lambda x: [float(v) for v in x], array)
            data = np.asarray(data)
        else:
            data = np.zeros([total_length, 4])
        
        encode_data = np.loadtxt(file_path)
        data = np.concatenate((data, encode_data), axis = 1)
        data_x = data.reshape((batch_size, total_length, input_size))
        data_y = np.roll(data, 1, axis = 0)
        act = data_y
        data_y = data_y.reshape((batch_size, total_length, input_size))
        x[batch] = data_x
        y[batch] = data_y
        actual[batch] = act
    
    print x.shape
    print y.shape
    print actual.shape
    
    return(x, y, actual)

def updataData(x, pred):
    y = x
    pred = np.asarray(pred)
    pred_x = np.roll(pred, -1, axis = 0)
    pred_x = pred_x.reshape((batch_size, total_length, input_size))
    pred_y = pred.reshape((batch_size, total_length, input_size))
#    print x.shape
#    print pred.shape
    for i in range(4):
        x[0, :, i] = pred_x[0, :, i]
        y[0, :, i] = pred_y[0, :, i]
    return(x, y)

def test(data_set):
    x, y, actual = readData(data_set, False)
    inH = np.zeros([num_batch, cell_size * num_layer])
    start_index = 0
    pred_series = []
    while start_index < total_length:
        print(start_index)
        X = x[:, start_index: start_index + seq_len, :]
        Y_ = y[:, start_index: start_index + seq_len, :]
        dic = {x_placeholder: X, y_placeholder: Y_, Hin: inH}
        l, outH, pred = sess.run([loss, state, Y], feed_dict = dic)
        start_index += seq_len
        print(l)
        for i in range(seq_len):
            pred_series.append(pred[i, :])
    plot(None, pred_series, actual, True)

def plot(loss_list, prediction_series, actual_series, need_press):
    print 'p'
    p = plt.subplot(2, 3, 1)
    plt.cla()
    if loss_list is not None:
        plt.plot(loss_list)
    p.set_title("Loss function")
    
    for iterator in range(4):
        p = plt.subplot(2, 3, iterator + 3)
        plt.cla()
        plt.axis([0, total_length, -1, 1])
        p.plot(np.asarray(prediction_series)[:, iterator], color = 'r')
        p.plot(np.asarray(actual_series)[:, iterator], color = 'b')
        title = "Joint info #" + str(iterator + 1)
        p.set_title(title)
    
    plt.draw()
    if need_press:
        plt.waitforbuttonpress()
    else:
        plt.pause(0.00001)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    x_b, y_b, actual_b = readData()
    loss_list = []
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in range(num_batch):
            x = x_b[batch]
            y = y_b[batch]
            actual = actual_b[batch]
            inH = np.zeros([num_batch, cell_size * num_layer])
            start_index = 0
            pred_series = []
            while start_index < total_length:
                X = x[:, start_index: start_index + seq_len, :]
                Y_ = y[:, start_indexL start_index + seq_len, :]
                dic = {x_placeholder: X, y_placeholder: Y_, Hin: inH}
                _, l, outH, pred = sess.run([optimizer, loss, state, Y], feed_dict = dic)
                inH = outH
                epoch_loss += l
                start_index += seq_len
                for i in range(seq_len):
                    pred_series.append(pred[i, :])
            # is not self, update joint
            if batch != 0:
                x_b[batch], y_b[batch] = updateData(x, pred_series)
        loss_list.append(epoch_loss)
        print("Epoch ", epoch, " loss: ", epoch_loss)
        if epoch % 5 == 0:
            plot(loss_list)

