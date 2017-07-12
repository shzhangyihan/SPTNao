import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import csv
import cv2

np.set_printoptions(threshold = 'nan')

# hyperparameters
learning_rate = 0.0005
num_epochs = 100
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

def initJoint():
    joint = np.zeros([batch_size, seq_len, 4]).astype(float)
    
    f = open(joint_path, 'rb')
    dataReader = csv.reader(f)
    for row in dataReader:
        j = np.asarray(row)
        break
    joint[:, :, 0:4] = j
    return joint

# GRU graph input
x_placeholder = tf.placeholder(tf.float64, [batch_size, seq_len, 40])
y_placeholder = tf.placeholder(tf.float64, [batch_size, seq_len, output_size])
Y_ = tf.reshape(y_placeholder, [-1, output_size])
Hin = tf.placeholder(tf.float64, [batch_size, cell_size * num_layer])
Xj = tf.Variable(initJoint())
reset = tf.assign(Xj, initJoint())

# GRU model
x_in = tf.concat([Xj, x_placeholder], 2)
mcell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(cell_size) for _ in range(num_layer)], state_is_tuple = False)
out, state = tf.nn.dynamic_rnn(mcell, x_placeholder, initial_state = Hin)

# output
Yj = tf.contrib.layers.linear(tf.reshape(out, [-1, cell_size]), 4)
Yv = tf.contrib.layers.linear(tf.reshape(out, [-1, cell_size]), 40)
Y = tf.concat([Yj, Yv], 1)
Xj = tf.reshape(Yj, [batch_size, seq_len, 4])

# loss and optimizer for self
loss = tf.contrib.losses.mean_squared_error(Y, Y_)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# loss and optimizer for other
Y_j, Y_v = tf.split(Y_, [4, 40], 1)
loss_other = tf.contrib.losses.mean_squared_error(Yv, Y_v)
optimizer_other = tf.train.AdamOptimizer(learning_rate).minimize(loss_other)

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

def plot(loss_list, x_b, y_b, actual_b, need_press):
    p = plt.subplot(3, 3, 1)
    plt.cla()
    plt.plot(loss_list)
    for batch in range(num_batch):
        x = x_b[batch]
        y = y_b[batch]
        inH = np.zeros([batch_size, cell_size * num_layer])
        start_index = 0
        pred_series = []
        sess.run(reset)
        while start_index < total_length:
            X = x[:, start_index: start_index + seq_len, 4:]
            Y_ = y[:, start_index: start_index + seq_len, :]
            dic = {x_placeholder: X, y_placeholder: Y_, Hin: inH}
            outH, pred = sess.run([state, Y], feed_dict = dic)
            inH = outH
            start_index += seq_len
            for i in range(seq_len):
                pred_series.append(pred[i, :])
        for iterator in range(4):
            p = plt.subplot(3, 3, batch * 4 + iterator + 2)
            plt.cla()
            plt.axis([0, total_length, -0.5, 1.5])
            p.plot(np.asarray(pred_series)[:, iterator], color = 'r')
            p.plot(np.asarray(actual_b[0])[:, iterator], color = 'b')
            title = "Batch " + str(batch) + " Joint " + str(iterator + 1)
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
            inH = np.zeros([batch_size, cell_size * num_layer])
            start_index = 0
            pred_series = []
            sess.run(reset)
            while start_index < total_length:
                X = x[:, start_index: start_index + seq_len, 4:]
                Y_ = y[:, start_index: start_index + seq_len, :]
                dic = {x_placeholder: X, y_placeholder: Y_, Hin: inH}
                # use self optimizer and loss if is self
                if batch == 0:
                    _, l, outH, pred = sess.run([optimizer, loss, state, Y], feed_dict = dic)
                else:
                    _, l, outH, pred = sess.run([optimizer_other, loss_other, state, Y], feed_dict = dic)
                inH = outH
                epoch_loss += l
                start_index += seq_len
                for i in range(seq_len):
                    pred_series.append(pred[i, :])
        loss_list.append(epoch_loss)
        print("Epoch ", epoch, " loss: ", epoch_loss)
        if epoch % 5 == 0:
            plot(loss_list, x_b, y_b, actual_b, False)
        if epoch == num_epochs - 1:
            plot(loss_list, x_b, y_b, actual_b, True)

