import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Hyper parameters
learning_rate = 0.01
training_epochs = 40
batch_size = 9
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
    layer_6 = tf.nn.sigmoid(tf.add(tf.matmul(layer_5, weights['decoder_h6']), 
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

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    
    # Training cycle
    for epoch in range(training_epochs):
        # Loop over all batches
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
