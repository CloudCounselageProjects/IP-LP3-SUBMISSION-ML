import tensorflow as tf
import numpy as np

class CNN(object):
    
    def __init__(self, 
                 batch_size,
                 learning_rate,
                 shape,
                 num_classes):
        
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.shape = shape
        self.num_classes = num_classes
        
        self.strides_size = 2
        self.kernel_size = 2
        self.padding = 'SAME'
        
    def init_weights(self, shape):
        random_dist = tf.truncated_normal(shape, stddev = 0.1)
        return tf.Variable(random_dist)
    
    def init_bias(self, shape):
        bias = tf.constant(0.1, shape = [shape])
        return tf.Variable(bias)
        
    def conv_layer(self, 
                   x, 
                   shape,
                   name = 'conv_'):
        
        W = self.init_weights(shape)
        b = self.init_bias(shape[3])
        
        conv = tf.nn.conv2d(x, W, strides=[1, self.strides_size, self.strides_size, 1],
                            padding = self.padding)
        layer = tf.add(conv, b)
        act = tf.nn.relu(layer)
        
        return act
        
    def fc_layer(self, X, shape):
        
        input_size = int(X.get_shape()[1])
        W = self.init_weights([input_size, shape])
        b = self.init_bias(shape)
        
        return tf.matmul(X, W ) + b
    
    def neural_net(self, X, hold_prob = 0.5):
        
        with tf.variable_scope('convolution') as scope:
            
            conv1 = self.conv_layer(X, [3, 3, 3,  32])
            conv2 = self.conv_layer(conv1, [5,5, 32, 64])            
            
        with tf.variable_scope('pooling') as scope:
            
            pool1 = tf.nn.max_pool(conv2, 
                                   ksize = [1, self.kernel_size , self.kernel_size ,1], 
                                   strides=[1, self.strides_size, self.strides_size , 1],
                                   padding = 'VALID')
            
         
        with tf.variable_scope('fc_layer') as scope:
            
            flat = tf.reshape(pool1, [-1 , 8 * 8 * 64])
            dropout_flat = tf.nn.dropout(flat, keep_prob=hold_prob) 
            
            fc1 = self.fc_layer(dropout_flat, 1024)
            fc1 = tf.nn.relu(fc1)
            dropout_fc1 = tf.nn.dropout(fc1, keep_prob=hold_prob)
            
            fc2 = self.fc_layer(dropout_fc1, self.num_classes)
            #fc2 = tf.nn.relu(fc2)
         
        self.y_prob = fc2
        #self.y_prob = tf.nn.softmax(fc2, name = 'Y_prob')
            
    def train_model(self, y,):
        pass        
