# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import tensorflow as tf


def get_emo_sequence(root_dir, subj):
    """Get trial sequence for each emotion condition."""
    beh_dir = os.path.join(root_dir, 'beh')
    par_dir = os.path.join(root_dir, 'par', 'emo')
    # get run number for subject
    tag_list = os.listdir(beh_dir)
    tag_list = [line for line in tag_list if line[-3:]=='csv']
    run_num = len([line for line in tag_list if line.split('_')[2]==subj])
    # sequence var
    seq = {}
    for r in range(run_num):
        # dict for run `r+1`
        seq[r+1] = {}
        train_trial_file = os.path.join(par_dir, 'trial_seq_%s_train.txt'%(r+1))
        test_trial_file = os.path.join(par_dir, 'trial_seq_%s_test.txt'%(r+1))
        train_trials = open(train_trial_file, 'r').readlines()
        test_trials = open(test_trial_file, 'r').readlines()
        train_trials = [line.strip().split(',') for line in train_trials]
        test_trials = [line.strip().split(',') for line in test_trials]
        trial_info_f = os.path.join(beh_dir,'trial_tag_%s_run%s.csv'%(subj,r+1))
        trial_info = open(trial_info_f, 'r').readlines()
        trial_info.pop(0)
        trial_info = [line.strip().split(',') for line in trial_info]
        for train_idx in range(len(train_trials)):
            img = train_trials[train_idx][1].split('\\')[1]
            emo = int([line[1] for line in trial_info if line[0]==img][0])
            if not emo in seq[r+1]:
                seq[r+1][emo] = {'train': [train_idx]}
            else:
                seq[r+1][emo]['train'].append(train_idx)
        for test_idx in range(len(test_trials)):
            img = test_trials[test_idx][1].split('\\')[1]
            emo = int([line[1] for line in trial_info if line[0]==img][0])
            if not 'test' in seq[r+1][emo]:
                seq[r+1][emo]['test'] = [test_idx]
            else:
                seq[r+1][emo]['test'].append(test_idx)
    return seq

def get_roi_ts(root_dir, seq):
    """Get neural activity time course of each roi on each emotion condition."""
    nii_dir = os.path.join(root_dir, 'nii')
    ppi_dir = os.path.join(root_dir, 'ppi')
    # load roi
    rois = nib.load(os.path.join(root_dir, 'group-level', 'rois', 'neurosynth',
                                 'cube_rois_r2.nii.gz')).get_data()
    #rois = nib.load(os.path.join(ppi_dir, 'cube_rois.nii.gz')).get_data()
    roi_num = int(rois.max())
    # get run info from scanlist
    scanlist_file = os.path.join(root_dir, 'doc', 'scanlist.csv')
    [scan_info, subj_list] = pyunpack.readscanlist(scanlist_file)
    for subj in subj_list:
        sid = subj.sess_ID
        print sid
        subj_dir = os.path.join(nii_dir, sid, 'emo')
        # get par index for each emo run
        if not 'emo' in subj.run_info:
            continue
        [run_idx, par_idx] = subj.getruninfo('emo')
        for i in range(10):
            if str(i+1) in par_idx:
                print 'Run %s'%(i+1)
                # load cope data
                ipar = par_idx.index(str(i+1))
                run_dir = os.path.join(subj_dir, '00'+run_idx[ipar])
                print run_dir
                train_cope_f = os.path.join(run_dir, 'train_merged_cope.nii.gz')
                test_cope_f = os.path.join(run_dir, 'test_merged_cope.nii.gz')
                train_cope = nib.load(train_cope_f).get_data()
                test_cope = nib.load(test_cope_f).get_data()
                # get trial sequence for each emotion
                for j in range(4):
                    seq_info = seq[i+1][j+1]
                    emo_data = np.zeros((91, 109, 91,
                                len(seq_info['train'])+len(seq_info['test'])))
                    emo_data[..., :len(seq_info['train'])] = train_cope[..., seq_info['train']]
                    emo_data[..., len(seq_info['train']):] = test_cope[..., seq_info['test']]
                    # get time course for each roi
                    roi_ts = np.zeros((emo_data.shape[3], roi_num))
                    for k in range(roi_num):
                        roi_ts[:, k] = niroi.extract_mean_ts(emo_data, rois==(k+1))
                    outfile = '%s_roi_ts_run%s_emo%s.npy'%(sid[:2], i+1, j+1)
                    outfile = os.path.join(ppi_dir, 'decovPPI', outfile)
                    np.save(outfile, roi_ts)

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
        for i in range(10000):
            # data preparation
            idx0 = np.arange(train_x.shape[0])
            np.random.shuffle(idx0)
            shuffle_train_x = train_x[idx0]
            shuffle_train_y = train_y[idx0]
            batch_x = shuffle_train_x[:50]
            batch_y = shuffle_train_y[:50]
            if i%100==0:
                train_accuracy = accuracy.eval(feed_dict={x: batch_x,
                                                          y_: batch_y})
                print "step %d, training accuracy %g"%(i, train_accuracy)
            train_step.run(feed_dict={x: batch_x, y_: batch_y})
        print accuracy.eval(feed_dict={x: test_x, y_: test_y})

if __name__=='__main__':
    db_dir = r'/Users/sealhuang/project/rois_meta_r2'
    x, y = load_data(db_dir, 'S1', one_hot=True)
    idx0 = np.arange(x.shape[0])
    np.random.shuffle(idx0)
    x = x[idx0]
    y = y[idx0]
    train_x = x[:int(x.shape[0]*0.9)]
    train_y = y[:int(x.shape[0]*0.9)]
    test_x = x[int(x.shape[0]*0.9):]
    test_y = y[int(x.shape[0]*0.9):]
    cls(train_x, train_y, test_x, test_y)


