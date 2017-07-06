import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import csv
import cv2

np.set_printoptions(threshold = 'nan')

# hyperparameters
learning_rate = 0.0005
num_epochs = 120
num_epochs_other = 200
total_length = 840
input_size = 4 + 40
output_size = input_size
cell_size = 4 * input_size
num_layer = 4
num_batch = 1
seq_len = 70

root_path = '../Nao_data/resize_4bit_noisy_40/'
joint_path = '../Nao_data/nao_data_joint.csv'

# GRU graph input
x_placeholder = tf.placeholder(tf.float32, [None, None, input_size])
y_placeholder = tf.placeholder(tf.float32, [None, None, output_size])
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

def readData(encode_num, is_self):
    file_path = root_path + 'encode_' + str(encode_num) + '.txt'
    print 'file = ', file_path
    array = []
    f = open(joint_path, 'rb')
    dataReader = csv.reader(f)
    for row in dataReader:
        array.append(row)
    data = map(lambda x:[float(v) for v in x], array)
    data = np.asarray(data)
    
    data_zero = np.zeros([total_length, 4])
    
    encode_data = np.loadtxt(file_path)
    data = np.concatenate((data, encode_data), axis = 1)
    data_zero = np.concatenate((data_zero, encode_data), axis = 1)
    if is_self:
        data_x = data
        data_y = np.roll(data, 1, axis = 0)
        actual = data_y
    else:
        data_x = data_zero
        data_y = np.roll(data_zero, 1, axis = 0)
        actual = np.roll(data, 1, axis = 0)
    
    data_x = data_x.reshape((num_batch, -1, input_size))
    data_y = data_y.reshape((num_batch, -1, input_size))
    
    return(data_x, data_y, actual)

def updataData(x, pred):
    y = x
    pred = np.asarray(pred)
    pred_x = np.roll(pred, -1, axis = 0)
    pred_x = pred_x.reshape((num_batch, total_length, input_size))
    pred_y = pred.reshape((num_batch, total_length, input_size))
#    print x.shape
#    print pred.shape
    for i in range(4):
        x[0, :, i] = pred_x[0, :, i]
        y[0, :, i] = pred_y[0, :, i]
    return(x, y)

def test(data_set):
    if data_set == 8:
        x, y, actual = readData(data_set, True)
    else:
        x, y, actual = readData(data_set, False)
    
    inH = np.zeros([num_batch, cell_size * num_layer])
    start_index = 0
    pred_series = []
    while start_index < total_length:
        X = x[:, start_index: start_index + 1, :]
        Y_ = y[:, start_index: start_index + 1, :]
        dic = {x_placeholder: X, y_placeholder: Y_, Hin: inH}
        outH, pred = sess.run([state, Y], feed_dict = dic)
        inH = outH
        start_index += 1
        pred_series.append(pred[0, :])
        if start_index < total_length:
            for i in range(4):
                x[0, start_index, i] = pred[0, i]
        
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
    
    x, y, actual = readData(8, True)
    loss_list = []
    for epoch in range(num_epochs):
        inH = np.zeros([num_batch, cell_size * num_layer])
        epoch_loss = 0
        start_index = 0
        pred_series = []
        while start_index < total_length:
            X = x[:, start_index: start_index + seq_len, :]
            Y_ = y[:, start_index: start_index + seq_len, :]
            dic = {x_placeholder: X, y_placeholder: Y_, Hin: inH}
            _, l, outH, pred = sess.run([optimizer, loss, state, Y], feed_dict = dic)
            inH = outH
            epoch_loss += l
            start_index += seq_len
            # prepare for plot
            if epoch % 5 == 0:
                for i in range(seq_len):
                    pred_series.append(pred[i, :])
        
        loss_list.append(epoch_loss)
        print("Epoch ", epoch, " loss: ", epoch_loss)
        if epoch % 5 == 0:
            plot(loss_list, pred_series, actual, False)
    
    test(8)
    test(0)
    test(1)
    test(2)
    test(3)
    test(4)
    test(5)
    test(6)
    test(7)
    
    x, y, actual = readData(0, False)
    for epoch in range(num_epochs_other):
        print x
        inH = np.zeros([num_batch, cell_size * num_layer])
        epoch_loss = 0
        start_index = 0
        pred_series = []
        while start_index < total_length:
            X = x[:, start_index: start_index + seq_len, :]
            Y_ = y[:, start_index: start_index + seq_len, :]
            dic = {x_placeholder: X, y_placeholder: Y_, Hin: inH}
            _, l, outH, pred = sess.run([optimizer, loss, state, Y], feed_dict = dic)
            inH = outH
            epoch_loss += l
            start_index += seq_len
            for i in range(seq_len):
                pred_series.append(pred[i, :])
                
        loss_list.append(epoch_loss)
        print("Epoch ", epoch, " loss: ", epoch_loss)
        if num_epochs_other % 5 == 0:
            plot(loss_list, pred_series, actual, False)
        if num_epochs_other == num_epochs_other - 1:
            plot(loss_list, pred_series, actual, True)
        
        x, y = updataData(x, pred_series)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

