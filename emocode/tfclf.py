# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import tensorflow as tf


def load_data(db_dir, subj_id, one_hot=True):
    """Load brain activity evoked by each stimulus and the corresponding
    emotion label from `subj_id`.
    """
    x = None
    y = None
    for i in range(10):
        tmp_x = None
        tmp_y = None
        for j in range(4):
            ts_file = os.path.join(db_dir,
                    '%s_roi_ts_run%s_emo%s.npy'%(subj_id, i+1, j+1))
            if os.path.exists(ts_file):
                ts = np.load(ts_file)
                if one_hot:
                    label = np.zeros((ts.shape[0], 4))
                    label[:, j] = 1
                else:
                    label = ones(ts.shape[0])*(j+1)
                if not isinstance(tmp_x, np.ndarray):
                    tmp_x = ts
                    tmp_y = label
                else:
                    tmp_x = np.concatenate((tmp_x, ts), axis=0)
                    tmp_y = np.concatenate((tmp_y, label), axis=0)
            else:
                print 'File %s does not exist'%(ts_file)
        m = tmp_x.mean(axis=0, keepdims=True)
        s = tmp_x.std(axis=0, keepdims=True)
        tmp_x = (tmp_x - m) / (s + 1e-5)
        if not isinstance(x, np.ndarray):
            x = tmp_x
            y = tmp_y
        else:
            x = np.concatenate((x, tmp_x), axis=0)
            y = np.concatenate((y, tmp_y), axis=0)
    return x, y

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
        batch_size = 50
        index_in_epoch = 0
        epochs_completed = 0
        for i in range(10000):
            start = index_in_epoch
            if epochs_completed==0 and start==0:
                perm0 = np.arange(train_x.shape[0])
                np.random.shuffle(perm0)
                shuffle_train_x = train_x[perm0]
                shuffle_train_y = train_y[perm0]
            # go to the next epoch
            if start + batch_size > train_x.shape[0]:
                # finish epoch
                epochs_completed += 1
                # get the rest examples in this epoch
                rest_num_examples = int(train_x.shape[0]) - start
                x_rest_part = shuffle_train_x[start:train_x.shape[0]]
                y_rest_part = shuffle_train_y[start:train_x.shape[0]]
                # shuffle the data
                perm = np.arange(train_x.shape[0])
                np.random.shuffle(perm)
                shuffle_train_x = train_x[perm]
                shuffle_train_y = train_y[perm]
                # start next epoch
                start = 0
                index_in_epoch = batch_size - rest_num_examples
                end = index_in_epoch
                x_new_part = shuffle_train_x[start:end]
                y_new_part = shuffle_train_y[start:end]
                batch = [np.concatenate((x_rest_part, x_new_part), axis=0),
                         np.concatenate((y_rest_part, y_new_part), axis=0)]
            else:
                index_in_epoch += batch_size
                end = index_in_epoch
                batch = [shuffle_train_x[start:end], shuffle_train_y[start:end]]
            # print training accuracy
            if i%100==0:
                train_accuracy = accuracy.eval(feed_dict={x: batch[0],
                                                          y_: batch[1]})
                print "step %d, training accuracy %g"%(i, train_accuracy)
            train_step.run(feed_dict={x: batch[0], y_: batch[1]})
        print accuracy.eval(feed_dict={x: test_x, y_: test_y})

if __name__=='__main__':
    db_dir = r'/Users/sealhuang/project/rois_meta_r2'
    x, y = load_data(db_dir, 'S1', one_hot=True)
    cls(train_x, train_y, test_x, test_y)


