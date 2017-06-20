import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

np.set_printoptions(threshold='nan')

# Hyper parameters
learning_rate = 0.1
training_epochs = 200
batch_size = 840
total_batch = 9
display_step = 1
examples_to_show = 10

# Network parameters
n_hidden_6 = 50
n_hidden_5 = 100
n_hidden_4 = 200
n_hidden_3 = 500
n_hidden_2 = 1000
n_hidden_1 = 2000
n_input = 4800

# data dir path
dir_path = {
    0: '../Nao_data/resized_img/Degree_0',
    1: '../Nao_data/resized_img/Degree_45',
    2: '../Nao_data/resized_img/Degree_90',
    3: '../Nao_data/resized_img/Degree_135',
    4: '../Nao_data/resized_img/Degree_180',
    5: '../Nao_data/resized_img/Degree_225',
    6: '../Nao_data/resized_img/Degree_270',
    7: '../Nao_data/resized_img/Degree_315',
    8: '../Nao_data/resized_img/self',
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
    # Encoder Hidden layer with relu activation #1
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['encoder_h1']), 
                                   biases['encoder_b1']))
    # Encoder Hidden layer with relu activation #2
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['encoder_h2']), 
                                   biases['encoder_b2']))
    # Encoder Hidden layer with relu activation #3
    layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weights['encoder_h3']), 
                                   biases['encoder_b3']))
    # Encoder Hidden layer with relu activation #4
    layer_4 = tf.nn.relu(tf.add(tf.matmul(layer_3, weights['encoder_h4']), 
                                   biases['encoder_b4']))
    # Encoder Hidden layer with relu activation #5
    layer_5 = tf.nn.relu(tf.add(tf.matmul(layer_4, weights['encoder_h5']), 
                                   biases['encoder_b5']))
    # Encoder Hidden layer with relu activation #6
    layer_6 = tf.nn.relu(tf.add(tf.matmul(layer_5, weights['encoder_h6']), 
                                   biases['encoder_b6']))
    return layer_6

# Building the decoder
def decoder(x):
    # Decoder Hidden layer with relu activation #1
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['decoder_h1']), 
                                   biases['decoder_b1']))
    # Decoder Hidden layer with relu activation #2
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['decoder_h2']), 
                                   biases['decoder_b2']))
    # Decoder Hidden layer with relu activation #3
    layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weights['decoder_h3']), 
                                   biases['decoder_b3']))
    # Decoder Hidden layer with relu activation #4
    layer_4 = tf.nn.relu(tf.add(tf.matmul(layer_3, weights['decoder_h4']), 
                                   biases['decoder_b4']))
    # Decoder Hidden layer with relu activation #5
    layer_5 = tf.nn.relu(tf.add(tf.matmul(layer_4, weights['decoder_h5']), 
                                   biases['decoder_b5']))
    # Decoder Hidden layer with relu activation #6
    layer_6 = tf.nn.relu(tf.add(tf.matmul(layer_5, weights['decoder_h6']), 
                                   biases['decoder_b6']))
    return layer_6

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
y_true = X

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# img loader
def load_img(batch, num):
    path = dir_path[batch]
    array = []
    for i in range(num):
        img_path = path + "/img_" + str(i+1) + ".png"
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_data = img.flatten()
        #print(img_data.shape)
        #print(img_data)
        array.append(img_data)
    data = np.asarray(array).reshape((-1, n_input))
    return data

# testing function
def test():
    f, a = plt.subplots(1, 2)
    batch_x = load_img(0, 1)
    result = sess.run(y_pred, feed_dict = {X: batch_x})
    print(batch_x[0])
    print(result[0])
    a[0].imshow(np.reshape(batch_x[0], (60, 80)))
    a[1].imshow(np.reshape(result[0], (60, 80)))
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
            _, c = sess.run([optimizer, cost], feed_dict = {X: batch_x})
            print("epoch: ", epoch, " batch: ", batch, " cost: ", c)
            epoch_loss = epoch_loss + c
        loss_list.append(epoch_loss)
        plt.cla()
        plt.plot(loss_list)
        plt.draw()
        plt.pause(0.00001)
    
    print("Optimization Finished!")
    test()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
