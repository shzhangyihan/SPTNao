import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import csv

np.set_printoptions(threshold='nan')

# Hyper parameters
learning_rate = 0.0004
training_epochs = 7000
batch_size = 840
total_batch = 9
corruption = 0.3

# Network parameters
n_hidden_6 = 40
n_hidden_5 = 100
n_hidden_4 = 250
n_hidden_3 = 500
n_hidden_2 = 800
n_hidden_1 = 1000
n_input = 1200

# data dir path
dir_path = {
    0: './Degree_0',
    1: './Degree_45',
    2: './Degree_90',
    3: './Degree_135',
    4: './Degree_180',
    5: './Degree_225',
    6: './Degree_270',
    7: './Degree_315',
    8: './self',
}

# tf Graph input 
X = tf.placeholder("float", [None, n_input])
mask = tf.placeholder("float", [None, n_input])

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
def encoder(x, mask):
    # corruption
    x_c = x * mask
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x_c, weights['encoder_h1']), 
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
encoder_op = encoder(X, mask)
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
#        img = np.divide(img, 32)
#        img = np.multiply(img, 32)
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
    mask_np = np.asfarray(np.ones(batch_x.shape))
    #mask_np = np.asfarray(np.random.binomial(1, 1-corruption, batch_x.shape))
    result, encode = sess.run([y_pred, encoder_op], feed_dict = {X: batch_x, mask: mask_np})
    print(batch_x[0])
    print(result[0])
    print(encode[0])
    
    encode_array = []
    encode_array.append(encode[0])
    fi = open('encode.csv', 'w')
    writer = csv.writer(fi, lineterminator='\n')
    writer.writerows(encode_array)
    img = batch_x[0] * mask_np
    a[0].imshow(np.reshape(img, (30, 40)), cmap='gray')
    a[1].imshow(np.reshape(result[0], (30, 40)), cmap='gray')
    f.show()
    plt.draw()
#    plt.waitforbuttonpress()

def save_encode():
    for batch in range(total_batch):
        batch_x = load_img(batch, batch_size)
        mask_np = np.asfarray(np.ones(batch_x.shape))
        result, encode, loss = sess.run([y_pred, encoder_op, cost], feed_dict = {X: batch_x, mask: mask_np})
        file_path = "./encode_" + str(batch) + ".txt"
        np.savetxt(file_path, encode)
        for i in range(batch_size):
            img_path = dir_path[batch] + "/decode/img_" + str(i+1) + ".png"
            img = np.reshape(result[i], (30, 40))
            img = np.asarray(img)
#            img = map(lambda x: x*255, img)
            img = np.multiply(img, 255)
#            print(img)
            cv2.imwrite(img_path, img)
        print("Batch " + str(batch) + " loss: " + str(loss))


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
            mask_np = np.asfarray(np.random.binomial(1, 1-corruption, batch_x.shape))
            _, c = sess.run([optimizer, cost], feed_dict = {X: batch_x, mask: mask_np})
            print("epoch: ", epoch, " batch: ", batch, " cost: ", c)
            epoch_loss = epoch_loss + c
        loss_list.append(epoch_loss)
        if epoch % 20 == 0:
            plt.cla()
            plt.plot(loss_list)
            plt.draw()
            plt.pause(0.00001)
        if epoch % 500 == 0:
            save_encode()
    
    print("Optimization Finished!")
    plt.cla()
    plt.plot(loss_list)
    plt.draw()
    plt.pause(0.00001)
    test()
    save_encode()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
