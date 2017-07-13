import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import csv

np.set_printoptions(threshold='nan')

# Hyper parameters
learning_rate = 0.0005
training_epochs = 12000
batch_size = 300
total_batch = 1

# Network parameters
n_hidden_6 = 100
n_hidden_5 = 200
n_hidden_4 = 400
n_hidden_3 = 800
n_hidden_2 = 1000
n_hidden_1 = 1400
n_input = 1200

# data dir path
dir_path = {
    0: './',
    1: '../Nao_data/processed_img/Degree_45',
    2: '../Nao_data/processed_img/Degree_90',
    3: '../Nao_data/processed_img/Degree_135',
    4: '../Nao_data/processed_img/Degree_180',
    5: '../Nao_data/processed_img/Degree_225',
    6: '../Nao_data/processed_img/Degree_270',
    7: '../Nao_data/processed_img/Degree_315',
    8: '../Nao_data/processed_img/self',
}

# tf Graph input 
X = tf.placeholder("float", [None, n_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'encoder_h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'encoder_h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),
    'encoder_h5': tf.Variable(tf.random_normal([n_hidden_4, n_hidden_5])),
    'encoder_h6': tf.Variable(tf.random_normal([n_hidden_5, n_hidden_6])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_6, n_hidden_5])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_5, n_hidden_4])),
    'decoder_h3': tf.Variable(tf.random_normal([n_hidden_4, n_hidden_3])),
    'decoder_h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_2])),
    'decoder_h5': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h6': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}

biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'encoder_b4': tf.Variable(tf.random_normal([n_hidden_4])),
    'encoder_b5': tf.Variable(tf.random_normal([n_hidden_5])),
    'encoder_b6': tf.Variable(tf.random_normal([n_hidden_6])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_5])),
    'decoder_b2': tf.Variable(tf.random_normal([n_hidden_4])),
    'decoder_b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'decoder_b4': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b5': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b6': tf.Variable(tf.random_normal([n_input])),
}

# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), 
                                   biases['encoder_b1']))
    # Encoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), 
                                   biases['encoder_b2']))
    # Encoder Hidden layer with sigmoid activation #3
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']), 
                                   biases['encoder_b3']))
    # Encoder Hidden layer with sigmoid activation #4
    layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['encoder_h4']), 
                                   biases['encoder_b4']))
    # Encoder Hidden layer with sigmoid activation #5
    layer_5 = tf.nn.sigmoid(tf.add(tf.matmul(layer_4, weights['encoder_h5']), 
                                   biases['encoder_b5']))
    # Encoder Hidden layer with sigmoid activation #6
    layer_6 = tf.nn.sigmoid(tf.add(tf.matmul(layer_5, weights['encoder_h6']), 
                                   biases['encoder_b6']))
    return layer_6

# Building the decoder
def decoder(x):
    # Decoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), 
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), 
                                   biases['decoder_b2']))
    # Decoder Hidden layer with sigmoid activation #3
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']), 
                                   biases['decoder_b3']))
    # Decoder Hidden layer with sigmoid activation #4
    layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['decoder_h4']), 
                                   biases['decoder_b4']))
    # Decoder Hidden layer with sigmoid activation #5
    layer_5 = tf.nn.sigmoid(tf.add(tf.matmul(layer_4, weights['decoder_h5']), 
                                   biases['decoder_b5']))
    # Decoder Hidden layer with sigmoid activation #6
    logit = tf.add(tf.matmul(layer_5, weights['decoder_h6']), biases['decoder_b6'])
    layer_6 = tf.nn.sigmoid(logit)
    
    return layer_6, logit

# Construct model
encoder_op = encoder(X)
decoder_op, logit = decoder(encoder_op)

# Prediction
y_pred = decoder_op
y_true = X

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
#optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
#cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = y_true, logits = logit))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# img loader
def load_img(batch, num):
    path = dir_path[batch]
    array = []
    for i in range(num):
        img_path = path + "/img_" + str(i+1) + ".png"
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#        img = np.add(img, 10)
        img = np.divide(img, 32)
        img = np.multiply(img, 32)
        img_data = img.flatten()
        img_data = np.asfarray(img_data)
#        img_data = np.add(img, 50.)
        img_data = map(lambda x: x/255, img_data)
        #print(img_data.shape)
        #print(img_data)
        array.append(img_data)
    data = np.asarray(array).reshape((-1, n_input))
    return data

# testing function
def test():
    f, a = plt.subplots(1, 2)
    batch_x = load_img(0, 1)
    result, encode = sess.run([y_pred, encoder_op], feed_dict = {X: batch_x})
    print(batch_x[0])
    print(result[0])
    print(encode[0])
    
    encode_array = []
    encode_array.append(encode[0])
    fi = open('encode.csv', 'w')
    writer = csv.writer(fi, lineterminator='\n')
    writer.writerows(encode_array)
    
    a[0].imshow(np.reshape(batch_x[0], (30, 40)), cmap='gray')
    a[1].imshow(np.reshape(result[0], (30, 40)), cmap='gray')
    f.show()
    plt.draw()
    plt.waitforbuttonpress()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    loss_list = []
    for epoch in range(training_epochs):
        # Loop over all batches
        epoch_loss = 0
        for batch in range(total_batch):
            batch_x = load_img(batch, batch_size)
            print(batch_x.shape)
            _, c = sess.run([optimizer, cost], feed_dict = {X: batch_x})
            print("epoch: ", epoch, " batch: ", batch, " cost: ", c)
            epoch_loss = epoch_loss + c
        loss_list.append(epoch_loss)
        if epoch % 20 == 0:
            plt.cla()
            plt.plot(loss_list)
            plt.draw()
            plt.pause(0.00001)
    
    print("Optimization Finished!")
    plt.cla()
    plt.plot(loss_list)
    plt.draw()
    plt.pause(0.00001)
    test()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
