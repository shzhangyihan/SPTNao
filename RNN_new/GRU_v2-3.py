import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import csv
import cv2
import time

# this version does a learning for both self and other
# has three RNN layers with size 80, 20, 80

np.set_printoptions(threshold = 'nan')

# hyperparameters
learning_rate = 0.0005
num_epochs = 150
total_length = 1200
joint_size = 8
vision_size = 30
input_size = joint_size + vision_size
output_size = input_size
cell_size = [100, 30, 100]
total_cell_size = np.sum(cell_size)
num_layer = 3
batch_size = 1
num_batch = 1
seq_len = 80

encode_path = './data/encode/'
joint_path = {
    0: './data/joint/nao_data_both.csv',
    1: './data/joint/nao_data_left.csv',
    2: './data/joint/nao_data_right.csv',
}
result_path = './result/'

def initJoint(length):
    joint = np.zeros([batch_size, length, joint_size]).astype(float)
    
    f = open(joint_path[0], 'rb')
    dataReader = csv.reader(f)
    for row in dataReader:
        j = np.asarray(row)
        break
    joint[:, :, 0:joint_size] = j
    return joint

with tf.variable_scope('GRU') as scope:
    # GRU graph input for training
    x_placeholder = tf.placeholder(tf.float64, [None, None, vision_size])
    y_placeholder = tf.placeholder(tf.float64, [None, None, output_size])
    Y_ = tf.reshape(y_placeholder, [-1, output_size])
    Hin = tf.placeholder(tf.float64, [None, total_cell_size])
    Xj = tf.Variable(initJoint(seq_len))
    reset = tf.assign(Xj, initJoint(seq_len))
    
    # GRU model for training
    mcell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(cell_size[i]) for i in range(num_layer)], state_is_tuple = False)
    x_in = tf.concat([Xj, x_placeholder], 2)
    out, state = tf.nn.dynamic_rnn(mcell, x_in, initial_state = Hin)
    
    # output
    Yj = tf.contrib.layers.fully_connected(tf.reshape(out, [-1, cell_size[2]]), joint_size, activation_fn = None, scope = "joint")
    Yv = tf.contrib.layers.fully_connected(tf.reshape(out, [-1, cell_size[2]]), vision_size, activation_fn = None, scope = "vision")
    Y = tf.concat([Yj, Yv], 1)
    Xj = tf.reshape(Yj, [batch_size, seq_len, joint_size])
    
    # loss and optimizer for self
    loss = tf.contrib.losses.mean_squared_error(Y, Y_)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    
    # loss and optimizer for other
    Y_j, Y_v = tf.split(Y_, [joint_size, vision_size], 1)
    loss_other = tf.contrib.losses.mean_squared_error(Yv, Y_v)
    optimizer_other = tf.train.AdamOptimizer(learning_rate).minimize(loss_other)
    
    # GRU graph input for testing
    x_t = tf.placeholder(tf.float64, [None, 1, vision_size])
    Hin_t = tf.placeholder(tf.float64, [None, total_cell_size])
    Xj_t = tf.Variable(initJoint(1))
    reset_t = tf.assign(Xj_t, initJoint(1))
    
    # GRU model for testing
    x_in_t = tf.concat([Xj_t, x_t], 2)
    scope.reuse_variables()
    out_t, state_t = tf.nn.dynamic_rnn(mcell, x_in_t, initial_state = Hin_t)
    
    # test output
    Yj_t = tf.contrib.layers.fully_connected(tf.reshape(out_t, [-1, cell_size[2]]), joint_size, activation_fn = None, reuse = True,scope = "joint")
    Yv_t = tf.contrib.layers.fully_connected(tf.reshape(out_t, [-1, cell_size[2]]), vision_size, activation_fn = None, reuse = True, scope = "vision")
    Y_t = tf.concat([Yj, Yv], 1)
    Xj_t = tf.reshape(Yj_t, [batch_size, 1, joint_size])

# run the tf graph on CPU
config = tf.ConfigProto(device_count = {'GPU': 0})

def readData(encode_num, is_self):
    file_path = encode_path + 'encode_' + str(encode_num) + '.txt'
    print 'file = ', file_path
    array = []
    f = open(joint_path[encode_num % 3], 'rb')
    dataReader = csv.reader(f)
    for row in dataReader:
        array.append(row)
    data = map(lambda x:[float(v) for v in x], array)
    data = np.asarray(data)
    
    data_zero = np.zeros([total_length, joint_size])
    
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
    
    data_x = data_x.reshape((batch_size, -1, input_size))
    data_y = data_y.reshape((batch_size, -1, input_size))
    
    return(data_x, data_y, actual)

def test(batch):
    x, y, actual = readData(batch, True)
    
    inH = np.zeros([batch_size, total_cell_size])
    sess.run(reset_t)
    seq_len = 1
    start_index = 0
    pred_series = []
    while start_index < total_length:
#        print start_index
        X = x[:, start_index: start_index + seq_len, joint_size:]
        Y_ = y[:, start_index: start_index + seq_len, :]
        dic = {x_t: X, Hin_t: inH}
        outH, pred = sess.run([state_t, Yj_t], feed_dict = dic)
        inH = outH
        start_index += seq_len
        for i in range(seq_len):
            pred_series.append(pred[i, :])
        
    plot(None, pred_series, actual, batch)

def plot(loss_list, prediction_series, actual_series, batch):
    print 'p'
    figure_path = result_path + 'figure_' + str(batch) + '.png'
    p = plt.subplot(3, 3, 1)
    plt.cla()
    if loss_list is not None:
        plt.plot(loss_list)
    p.set_title("Loss function")
    
    for iterator in range(joint_size):
        p = plt.subplot(3, 3, iterator + 2)
        plt.cla()
        plt.axis([0, total_length, -0.5, 1.5])
        p.plot(np.asarray(prediction_series)[:, iterator], color = 'r')
        p.plot(np.asarray(actual_series)[:, iterator], color = 'b')
        title = "Joint info #" + str(iterator + 1)
        p.set_title(title)
    
    plt.draw()
    plt.savefig(figure_path)

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    
    # load data for batches
    data_set = [21]
    x_b = np.zeros([num_batch, batch_size, total_length, input_size])
    y_b = np.zeros([num_batch, batch_size, total_length, input_size])
    actual_b = np.zeros([num_batch, total_length, input_size])
    
    for batch in range(num_batch):
        x_b[batch], y_b[batch], actual_b[batch] = readData(data_set[batch], True)
    
    loss_list = []
    time0 = time.time()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in range(num_batch):
            x = x_b[batch]
            y = y_b[batch]
            actual = actual_b[batch]
            inH = np.zeros([batch_size, total_cell_size])
            sess.run(reset)
            start_index = 0
            pred_series = []
            while start_index < total_length:
                X = x[:, start_index: start_index + seq_len, joint_size:]
                Y_ = y[:, start_index: start_index + seq_len, :]
                dic = {x_placeholder: X, y_placeholder: Y_, Hin: inH}
                # if self
                if data_set[batch] >= 21:
                    _, l, outH, pred = sess.run([optimizer, loss, state, Yj], feed_dict = dic)
                else:
                    _, l, outH, pred = sess.run([optimizer_other, loss_other, state, Yj], feed_dict = dic)
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
    test(21)
    test(22)
    test(23)

    
