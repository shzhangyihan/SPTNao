from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import csv

# hyperparameters
num_epochs = 200
total_series_length = 320
input_size = 8
output_size = input_size
cell_size = 4*input_size
batch_size = 320
num_batch = 1
num_layer = 2
seq_len = 20

# GRU define
# placeholders
x_placeholder = tf.placeholder(tf.float32, [num_batch, seq_len, input_size])
y_placeholder = tf.placeholder(tf.float32, [num_batch, seq_len, output_size])
Y_ = tf.reshape(y_placeholder, [-1, output_size])
Hin = tf.placeholder(tf.float32, [num_batch, cell_size * num_layer])
    
# GRU model
mcell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(cell_size) for _ in range(num_layer)], state_is_tuple = False)
Hr, H = tf.nn.dynamic_rnn(mcell, x_placeholder, initial_state = Hin)

# output layer
Hf = tf.reshape(Hr, [-1, cell_size])
Y = tf.contrib.layers.linear(Hf, output_size)

# loss and training step
loss = tf.contrib.losses.mean_squared_error(Y, Y_)
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

def readData():
    array = []
    f = open('../Nao_data/nao_data.csv', 'rb')
    dataReader = csv.reader(f)
    for row in dataReader:
        array.append(row)
    data = map(lambda x:[float(v) for v in x], array)
    
    datax = np.asarray(data)
    datay = np.roll(data, 1, axis = 0)
    
    actual = datay
    datax = datax.reshape((num_batch, -1, input_size))
    datay = datay.reshape((num_batch, -1, input_size))

    print("x shape: ", datax.shape)
    print("x = ", np.asarray(datax))
    print("y = ", np.asarray(datay))
    
    return(datax, datay, actual)

def plot(loss_list, prediction_series, actual_series):
    p = plt.subplot(3, 3, 1)
    plt.cla()
    plt.plot(loss_list)
    p.set_title("Loss function")
    
    for dimension in range(input_size):
        p = plt.subplot(3, 3, dimension + 2)
        plt.cla()
        plt.axis([0, total_series_length, 0, 1.5])
        p.plot(np.asarray(prediction_series)[:, dimension], color = 'r')
        p.plot(np.asarray(actual_series)[:, dimension], color = 'b')
        title = "Input dimension " + str(dimension)
        p.set_title(title)
    
    plt.draw()
    plt.pause(0.00001)

def train_neural_network():
    x, y, actual = readData()
    loss_list = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(num_epochs):
            inH = np.zeros([num_batch, cell_size * num_layer])
            epoch_loss = 0
            start_index = 0
            pred_series = []
            while start_index < total_series_length:
                X = x[:, start_index: start_index+seq_len, :]
                Y_ = y[:, start_index: start_index+seq_len, :]
                dic = {x_placeholder: X, y_placeholder:Y_, Hin:inH}
                # training start
                _, loss_, outH, pred = sess.run([train_step, loss, H, Y], feed_dict = dic)
                inH = outH
                epoch_loss += loss_
                start_index += seq_len
                if epoch % 5 == 0:
                    for i in range(seq_len):
                        pred_series.append(pred[i,:])
            # end of epoch
            loss_list.append(epoch_loss)
            print("Epoch ", epoch, "loss: ", epoch_loss)
            # plot
            if epoch % 5 == 0:
                print("pred", np.asarray(pred_series).shape)
                print("actu", np.asarray(actual).shape)
                plot(loss_list, pred_series, actual)

plt.ioff()
plt.show()
train_neural_network()


