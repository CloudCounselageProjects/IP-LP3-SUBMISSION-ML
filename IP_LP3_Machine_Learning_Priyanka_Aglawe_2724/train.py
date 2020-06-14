import tensorflow as tf
import numpy as np
import os

from model import *
from utils import *

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('restore_checkpoint', None,
                       'Give the fail path where the saved model is')
tf.flags.DEFINE_string('save_path', 'model',
                       'Give the fail path where the model should be saved')

epoch = 1001
batch_size = 117

X_train, X_test, Y_train, Y_test = read_img()

y_train = load_encoder_output(Y_train)
y_test = load_encoder_output(Y_test)

cnn = CNN(batch_size= 64, learning_rate=0.001, shape = [64,64,3],
          num_classes=13)

graph = tf.Graph()

with graph.as_default():
    
    X = tf.placeholder(tf.float32, shape = [None, 64, 64, 3])
    y = tf.placeholder(tf.float32, shape = [None, 13])
    hold_prob = tf.placeholder(tf.float32)
    
    cnn.neural_net(X, hold_prob)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,
                                                                           logits=cnn.y_prob))
    optimizer = tf.train.AdamOptimizer(learning_rate = cnn.learning_rate)
    loss = optimizer.minimize(cross_entropy)
    
    saver = tf.train.Saver()
    
    init = tf.global_variables_initializer()

tf.reset_default_graph()
    

if FLAGS.restore_checkpoint is not None:
    checkpoint = tf.train.get_checkpoint_state(os.path.join('checkpoints', FLAGS.restore_checkpoint))
    meta_graph_path = checkpoint.model_checkpoint_path + '.meta'
    restore = tf.train.import_meta_graph(meta_graph_path)
    step = int(meta_graph_path.split("-")[1].split(".")[0])
else:
    restore = None
    step = 0
    
with tf.Session(graph = graph) as sess:
    
    if restore is not None:
        restore.restore(sess, tf.train.latest_checkpoint(os.path.join('checkpoints', FLAGS.restore_checkpoint)))
        
    sess.run(init)
    for i in range(epoch):
        
        for batch in range(0, len(X_train), batch_size):
            sess.run(loss, feed_dict = {X : X_train[batch : batch + batch_size],
                                        y : y_train[batch : batch + batch_size],
                                        hold_prob : 0.5
                    })
    
        if i % 1000 == 0:
            print('Currently on step {}'.format(i))
            print('Accuracy is:')
            # Test the Train Model
            matches = tf.equal(tf.argmax(cnn.y_prob,1),tf.argmax(y,1))
            acc = tf.reduce_mean(tf.cast(matches,tf.float32))
            print(sess.run(acc,feed_dict = { X : X_train[batch : batch + batch_size],
                                        y : y_train[batch : batch + batch_size],
                                        hold_prob : 0.5 }))
     
            saver.save(sess, save_path = "./checkpoints/" + FLAGS.save_path + \
                           "/model.ckpt", global_step = step + (i * (batch + 1)))
