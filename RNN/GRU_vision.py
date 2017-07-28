import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import csv
import cv2
import time

# this version does a learning for visual prediction

np.set_printoptions(threshold = 'nan')

# hyperparameters
learning_rate = 0.0003
num_epochs = 300
total_length = 1200
vision_size = 30
input_size = vision_size
output_size = input_size
cell_size = [120, 40, 120]
total_cell_size = np.sum(cell_size)
num_layer = 3
batch_size = 1
num_batch = 8
seq_len = 80

encode_path = './data/encode/'
result_path = './result_vision/'

with tf.variable_scope('GRU') as scope:
    # GRU graph input for training
    x_placeholder = tf.placeholder(tf.float64, [None, None, vision_size])
    y_placeholder = tf.placeholder(tf.float64, [None, None, output_size])
    Y_ = tf.reshape(y_placeholder, [-1, output_size])
    Hin = tf.placeholder(tf.float64, [None, total_cell_size])
    
    # GRU model for training
    mcell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(cell_size[i]) for i in range(num_layer)], state_is_tuple = False)
    out, state = tf.nn.dynamic_rnn(mcell, x_placeholder, initial_state = Hin)
    
    # output
    Yv = tf.contrib.layers.fully_connected(tf.reshape(out, [-1, cell_size[2]]), vision_size, activation_fn = None, scope = "vision")
    
    # loss and optimizer for self
    loss = tf.contrib.losses.mean_squared_error(Yv, Y_)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# run the tf graph on CPU
config = tf.ConfigProto(device_count = {'GPU': 0})

def readData(encode_num):
    file_path = encode_path + 'encode_' + str(encode_num) + '.txt'
    print 'file = ', file_path
    
    encode_data = np.loadtxt(file_path)
    data_x = encode_data
    data_y = np.roll(data_x, 1, axis = 0)
    actual = data_y
    data_x = data_x.reshape((batch_size, -1, input_size))
    data_y = data_y.reshape((batch_size, -1, input_size))
    
    return(data_x, data_y, actual)

def test(batch, middle_list):
    x, y, actual = readData(batch)
    
    inH = np.zeros([batch_size, total_cell_size])
    seq_len = 1
    start_index = 0
    pred_series = []
    loss_list = []
    while start_index < total_length:
#        print start_index
        X = x[:, start_index: start_index + seq_len, :]
        Y_ = y[:, start_index: start_index + seq_len, :]
        dic = {x_placeholder: X, y_placeholder: Y_, Hin: inH}
        outH, pred, l = sess.run([state, Yv, loss], feed_dict = dic)
        inH = outH
        middle = np.zeros(cell_size[1] + 1)
        middle[0] = batch
        middle[1: cell_size[1] + 1] = outH[0, cell_size[0]: cell_size[0] + cell_size[1]]
        middle_list.append(middle)
        start_index += seq_len
        loss_list.append(l)
        for i in range(seq_len):
            pred_series.append(pred[i, :])
        
    plot(loss_list, pred_series, actual, batch)

def plot(loss_list, prediction_series, actual_series, batch):
    print 'p'
    figure_path = result_path + 'figure_' + str(batch) + '.png'
    p = plt.subplot(3, 3, 1)
    plt.cla()
    if loss_list is not None:
        plt.plot(loss_list)
    p.set_title("Loss function")
    
    for iterator in range(8):
        p = plt.subplot(3, 3, iterator + 2)
        plt.cla()
        plt.axis([0, total_length, -0.5, 1.5])
        p.plot(np.asarray(prediction_series)[:, iterator], color = 'r')
        p.plot(np.asarray(actual_series)[:, iterator], color = 'b')
        title = "Vision info #" + str(iterator + 1)
        p.set_title(title)
    
    plt.draw()
    plt.savefig(figure_path)

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    
    # load data for batches
    data_set = [0, 3, 6, 9, 12, 15, 18, 21]
    x_b = np.zeros([num_batch, batch_size, total_length, input_size])
    y_b = np.zeros([num_batch, batch_size, total_length, input_size])
    actual_b = np.zeros([num_batch, total_length, input_size])
    
    for batch in range(num_batch):
        x_b[batch], y_b[batch], actual_b[batch] = readData(data_set[batch])
    
    loss_list = []
    time0 = time.time()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in range(num_batch):
            x = x_b[batch]
            y = y_b[batch]
            actual = actual_b[batch]
            inH = np.zeros([batch_size, total_cell_size])
            start_index = 0
            pred_series = []
            while start_index < total_length:
                X = x[:, start_index: start_index + seq_len, :]
                Y_ = y[:, start_index: start_index + seq_len, :]
                dic = {x_placeholder: X, y_placeholder: Y_, Hin: inH}
                _, l, outH, pred = sess.run([optimizer, loss, state, Yv], feed_dict = dic)
                inH = outH
                epoch_loss += l
                start_index += seq_len
                # prepare for plotting
                if epoch % 5 == 0:
                    for i in range(seq_len):
                        pred_series.append(pred[i, :])
        
        loss_list.append(epoch_loss)
        print("Epoch ", epoch, " loss: ", epoch_loss)
        if epoch % 5 == 0:
            time1 = time.time()
            print "time diff = " + str(time1 - time0)
            plot(loss_list, pred_series, actual, 'train')
    
    time2 = time.time()
    print "time per step = " + str((time2 - time0) / num_epochs)
    
    # store the testing figure and middle layer
    middle_list = []
    for data in data_set:
        test(data, middle_list)
    print len(middle_list)
    
    f = open('./result_vision/middle_layer.csv', 'w')
    writer = csv.writer(f, lineterminator='\n')
    writer.writerows(middle_list)
    f.close()
    
