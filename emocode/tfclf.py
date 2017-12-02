# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import tensorflow as tf


def load_data(db_dir, subj_id, one_hot=True):
    """Load brain activity evoked by each stimulus and the corresponding
    emotion label from `subj_id`.
    """
    pass

def cls(train_x, train_y, test_x, test_y):
    """Emotion classifier based on softmax"""
    pred_var_num = train_x.shape[1]
    # graph init
    x = tf.placeholder("float", [None, pred_var_num])
    y_ = tf.placeholder("float", [None, 4])
    W = tf.Variable(tf.zeros([pred_var_num, 4]))
    b = tf.Variable(tf.zeros([4]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # run model
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for i in range(1000):
            # XXX
            batch = mnist.train.next_batch(50)
            if i%100==0:
                train_accuracy = accuracy.eval(feed_dict={
                                    x: batch[0], y_: batch[1]})
                print "step %d, training accuracy %g"%(i, train_accuracy)
            train_step.run(feed_dict={x: batch[0], y_: batch[1]})
        print accuracy.eval(feed_dict={x: test_x, y_: test_y})

if __name__=='__main__':
    pass


