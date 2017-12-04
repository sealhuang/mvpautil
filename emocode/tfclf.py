# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import tensorflow as tf


def load_data(db_dir, subj_id, one_hot=True):
    """Load brain activity evoked by each stimulus and the corresponding
    emotion label from `subj_id`.
    """
    train_x = None
    train_y = None
    test_x = None
    test_y = None
    for i in range(10):
        data_file = os.path.join(db_dir, '%s_run%s_roi_data.npz'%(subj_id, i+1))
        if os.path.exists(data_file):
            npz = np.load(data_file)
            x = np.concatenate((npz['arr_0'], npz['arr_2']), axis=0)
            m = x.mean(axis=0)
            s = x.std(axis=0)
            tmp_trn_x = (npz['arr_0'] - m) / (s + 1e-5)
            tmp_test_x = (npz['arr_2'] - m) / (s + 1e-5)
            if one_hot:
                tmp_trn_y = np.zeros((npz['arr_1'].shape[0], 4))
                tmp_trn_y[range(npz['arr_1'].shape[0]), npz['arr_1']-1] = 1
                tmp_test_y = np.zeros((npz['arr_3'].shape[0], 4))
                tmp_test_y[range(npz['arr_3'].shape[0]), npz['arr_3']-1] = 1
            else:
                tmp_trn_y = npz['arr_1']
                tmp_test_y = npz['arr_3']
            if not isinstance(train_x, np.ndarray):
                train_x = tmp_trn_x
                train_y = tmp_trn_y
                test_x = tmp_test_x
                test_y = tmp_test_y
            else:
                train_x = np.concatenate((train_x, tmp_trn_x), axis=0)
                train_y = np.concatenate((train_y, tmp_trn_y), axis=0)
                test_x = np.concatenate((test_x, tmp_test_x), axis=0)
                test_y = np.concatenate((test_y, tmp_test_y), axis=0)
        else:
            print 'File %s does not exist'%(data_file)
    return train_x, train_y, test_x, test_y

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
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # run model
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        batch_size = 30
        index_in_epoch = 0
        epochs_completed = 0
        for i in range(5000):
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
    train_x, train_y, test_x, test_y = load_data(db_dir, 'S7', one_hot=True)
    cls(train_x, train_y, test_x, test_y)

