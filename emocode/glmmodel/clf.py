# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import random
import nibabel as nib
from sklearn import svm

from nitools import roi as niroi
from nitools import base as nibase


def gen_func_mask(root_dir, sid):
    """Generate a functional mask based on reference frame from two sessions."""
    work_dir = os.path.join(root_dir, 'workshop', 'glmmodel', 'searchlight')
    db_dir = os.path.join(root_dir, 'workshop', 'glmmodel', 'nii')
    ref_1 = os.path.join(db_dir, sid, 'ref_vol_session1.nii.gz')
    ref_2 = os.path.join(db_dir, sid, 'ref_vol_session2_mcf.nii.gz')
    ref_1_data = nib.load(ref_1).get_data()
    ref_2_data = nib.load(ref_2).get_data()
    mean_data = (ref_1_data + ref_2_data) / 2
    mean_data = mean_data.reshape(-1)
    mean_data.sort()
    thresh = int(mean_data[int(np.around(mean_data.shape[0]*0.9))]*0.5)
    print 'Threshold for mean reference frame: %s'%(thresh)
    mask_file = os.path.join(work_dir, sid, 'func_mask.nii.gz')
    cmd_str = ['fslmaths', ref_1, '-add', ref_2, '-div', '2', '-thr',
               str(thresh), '-bin', mask_file]
    print ' '.join(cmd_str)
    os.system(' '.join(cmd_str))

def get_stimuli_label(root_dir, sid):
    """Get subject's trial tag for each run."""
    beh_dir = os.path.join(root_dir, 'beh')
    # get subject name
    subj_name = {'S1': 'liqing', 'S2': 'zhangjipeng', 'S3': 'zhangdan',
                 'S4': 'wanghuicui', 'S5': 'zhuzhiyuan', 'S6': 'longhailiang',
                 'S7': 'liranran'}
    subj = subj_name[sid]
    # stimuli label list var
    stim_label_list = []
    for i in range(10):
        stim_label = []
        img_list = []
        # load experiment record
        record = os.path.join(beh_dir, 'trial_record_%s_run%s.csv'%(subj, i+1))
        record_info = open(record, 'r').readlines()
        record_info.pop(0)
        record_info = [line.strip().split(',') for line in record_info]
        for line in record_info:
            if not line[0] in img_list:
                img_list.append(line[0])
                stim_label.append(int(line[1]))
        stim_label_list.append(stim_label)
    return stim_label_list

def get_roi_cope_mvps(cope_list, trial_tag_list, roi_coord):
    """Get MVPs from each nii file based on trial info."""
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    for i in range(len(cope_list)):
        cope_data = cope_list[i]
        trial_tag = trial_tag_list[i]
        for t in range(cope_data.shape[3]):
            vtr = niroi.get_voxel_value(roi_coord, cope_data[..., t])
            if t<72:
                train_x.append(vtr.tolist())
                train_y.append(trial_tag[t])
            else:
                test_x.append(vtr.tolist())
                test_y.append(trial_tag[t])
    return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)

def roi_svm(root_dir, sid, roi_file):
    """ROI based SVM classifcation."""
    #-- dir config
    beta_dir = os.path.join(root_dir, 'workshop', 'glmmodel', 'betas')
    work_dir = os.path.join(root_dir, 'workshop', 'glmmodel', 'roi_analysis')
    
    #-- read mask file
    print 'Load mask data ...'
    mask_file = os.path.join(work_dir, sid, roi_file)
    mask_data = nib.load(mask_file).get_data()
    roi_idx = np.unique(mask_data)
    
    #-- load estimated beta maps
    print 'Load estimated beta maps from training datasets ...'
    train_betas1_file=os.path.join(beta_dir,sid,'%s_beta_train_s1.nii.gz'%(sid))
    train_betas2_file=os.path.join(beta_dir,sid,'%s_beta_train_s2.nii.gz'%(sid))
    train_betas1 = nib.load(train_betas1_file).get_data()
    train_betas2 = nib.load(train_betas2_file).get_data()
    train_betas = np.concatenate((train_betas1, train_betas2), axis=3)
    print 'Load estimated beta maps from testing datasets ...'
    test_betas1_file = os.path.join(beta_dir, sid,'%s_beta_val_s1.nii.gz'%(sid))
    test_betas2_file = os.path.join(beta_dir, sid,'%s_beta_val_s2.nii.gz'%(sid))
    test_betas1 = nib.load(test_betas1_file).get_data()
    test_betas2 = nib.load(test_betas2_file).get_data()
    test_betas = np.concatenate((test_betas1, test_betas2), axis=3)
    # data normalization
    m = np.mean(train_betas, axis=3, keepdims=True)
    s = np.std(train_betas, axis=3, keepdims=True)
    train_betas = (train_betas - m) / (s + 1e-10)
    m = np.mean(test_betas, axis=3, keepdims=True)
    s = np.std(test_betas, axis=3, keepdims=True)
    test_betas = (test_betas - m) / (s + 1e-10)

    print train_betas.shape
    print test_betas.shape

    #-- get stimuli label info
    print 'Load stimuli label info ...'
    stim_label_list = get_stimuli_label(root_dir, sid)
    train_label = np.concatenate((stim_label_list[0], stim_label_list[1],
                                  stim_label_list[2], stim_label_list[3],
                                  stim_label_list[5], stim_label_list[6],
                                  stim_label_list[7], stim_label_list[8]))
    test_label = np.concatenate((stim_label_list[4], stim_label_list[9]))

    print train_label.shape
    print test_label.shape

    #-- svm-based classifier
    # for loop for each roi
    for i in roi_idx:
        if not i:
            continue
        print 'ROI %s'%(i)
        roi_mask = mask_data==i
        roi_coord = niroi.get_roi_coord(roi_mask)
        train_x = []
        test_x = []
        for t in range(train_betas.shape[3]):
            vtr = niroi.get_voxel_value(roi_coord, train_betas[..., t])
            train_x.append(vtr.tolist())
        for t in range(test_betas.shape[3]):
            vtr = niroi.get_voxel_value(roi_coord, test_betas[..., t])
            test_x.append(vtr.tolist())
        train_x = np.array(train_x)
        test_x = np.array(test_x)
        # classifier
        # kernel can be specified as linear, poly, rbf, and sigmod
        kernel = 'rbf'
        clf = svm.SVC(kernel=kernel)
        clf.fit(train_x, train_label)
        pred = clf.predict(test_x)
        for e in range(4):
            acc = np.sum(pred[test_label==(e+1)]==(e+1))*1.0 / np.sum(test_label==(e+1))
            print acc

def svm_searchlight(root_dir, sid, test_run_idx):
    """SVM based searchlight analysis."""
    print 'Searchlight analysis on Subject %s - test run %s'%(sid, test_run_idx)
    #-- dir config
    beta_dir = os.path.join(root_dir, 'workshop', 'glmmodel', 'betas')
    work_dir = os.path.join(root_dir, 'workshop', 'glmmodel', 'searchlight')
    
    #-- read mask file
    print 'Load mask data ...'
    mask_file = os.path.join(work_dir, sid, 'func_mask.nii.gz')
    mask_data = nib.load(mask_file).get_data()
    mask_data = mask_data>0
    
    #-- load estimated beta maps
    print 'Load estimated beta maps from training datasets ...'
    train_betas1_file = os.path.join(beta_dir, sid,
                            '%s_beta_train_s1_t%s.nii.gz'%(sid, test_run_idx))
    train_betas2_file = os.path.join(beta_dir, sid,
                            '%s_beta_train_s2_t%s.nii.gz'%(sid, test_run_idx))
    train_betas1 = nib.load(train_betas1_file).get_data()
    train_betas2 = nib.load(train_betas2_file).get_data()
    train_betas = np.concatenate((train_betas1, train_betas2), axis=3)
    print 'Load estimated beta maps from testing datasets ...'
    test_betas1_file = os.path.join(beta_dir, sid,
                            '%s_beta_val_s1_t%s.nii.gz'%(sid, test_run_idx))
    test_betas2_file = os.path.join(beta_dir, sid,
                            '%s_beta_val_s2_t%s.nii.gz'%(sid, test_run_idx))
    test_betas1 = nib.load(test_betas1_file).get_data()
    test_betas2 = nib.load(test_betas2_file).get_data()
    test_betas = np.concatenate((test_betas1, test_betas2), axis=3)
    # data normalization
    m = np.mean(train_betas, axis=3, keepdims=True)
    s = np.std(train_betas, axis=3, keepdims=True)
    train_betas = (train_betas - m) / (s + 1e-5)
    m = np.mean(test_betas, axis=3, keepdims=True)
    s = np.std(test_betas, axis=3, keepdims=True)
    test_betas = (test_betas - m) / (s + 1e-5)

    print train_betas.shape
    print test_betas.shape

    #-- get stimuli label info
    print 'Load stimuli label info ...'
    stim_label_list = get_stimuli_label(root_dir, sid)
    test_label = np.concatenate((stim_label_list[test_run_idx-1],
                                 stim_label_list[5+test_run_idx-1]))
    stim_label_list.pop(test_run_idx-1)
    stim_label_list.pop(5+test_run_idx-2)
    train_label = np.concatenate(tuple(item for item in stim_label_list))

    print train_label.shape
    print test_label.shape

    #-- svm-based searchlight
    clf_results = np.zeros((64, 64, 33, 4))
    # for loop for voxel-wise searchlight
    mask_coord = niroi.get_roi_coord(mask_data)
    ccount = 0
    for c in mask_coord:
        ccount += 1
        print ccount
        cube_roi = np.zeros((64, 64, 33))
        cube_roi = niroi.cube_roi(cube_roi, c[0], c[1], c[2], 2, 1)
        cube_coord = niroi.get_roi_coord(cube_roi)
        train_x = []
        test_x = []
        for t in range(train_betas.shape[3]):
            vtr = niroi.get_voxel_value(cube_coord, train_betas[..., t])
            train_x.append(vtr.tolist())
        for t in range(test_betas.shape[3]):
            vtr = niroi.get_voxel_value(cube_coord, test_betas[..., t])
            test_x.append(vtr.tolist())
        train_x = np.array(train_x)
        test_x = np.array(test_x)
        # classifier
        # kernel can be specified as linear, poly, rbf, and sigmod
        kernel = 'rbf'
        clf = svm.SVC(kernel=kernel)
        clf.fit(train_x, train_label)
        pred = clf.predict(test_x)
        for e in range(4):
            acc = np.sum(pred[test_label==(e+1)]==(e+1))*1.0 / np.sum(test_label==(e+1))
            print acc
            clf_results[c[0], c[1], c[2], e] = acc
    # save to nifti
    aff = nib.load(mask_file).affine
    result_file = os.path.join(work_dir, sid, 'svm_%s_t%s.nii.gz'%(kernel,
                                                                test_run_idx))
    nibase.save2nifti(clf_results, aff, result_file)
    func2anat_mat = os.path.join(root_dir, 'workshop', 'glmmodel', 'nii',
                                 sid, 'ref_vol2highres.mat')
    t1brain_vol = os.path.join(root_dir, 'nii', sid+'P1', '3danat',
                               'reg_fsl', 'T1_brain.nii.gz')
    if os.path.exists(func2anat_mat):
        result2_file = os.path.join(work_dir, sid,
                            'svm_%s_t%s_highres.nii.gz'%(kernel, test_run_idx))
        str_cmd = ['flirt', '-in', result_file, '-ref', t1brain_vol,
                   '-applyxfm', '-init', func2anat_mat, '-out', result2_file]
        os.system(' '.join(str_cmd))

def svm_searchlight_cv(root_dir, sid):
    """SVM based searchlight analysis in a cross-validation approach."""
    for r in range(1, 6):
        svm_searchlight(root_dir, sid, r)

def random_svm_searchlight(root_dir, sid, test_run_idx, rand_num):
    """Generate a NULL distribution for SVM based searchlight analysis."""
    print 'Searchlight analysis on Subject %s - test run %s'%(sid, test_run_idx)
    #-- dir config
    beta_dir = os.path.join(root_dir, 'workshop', 'glmmodel', 'betas')
    work_dir = os.path.join(root_dir, 'workshop', 'glmmodel', 'searchlight')
    
    #-- read mask file
    print 'Load mask data ...'
    mask_file = os.path.join(work_dir, sid, 'func_mask.nii.gz')
    mask_data = nib.load(mask_file).get_data()
    mask_data = mask_data>0
    
    #-- load estimated beta maps
    print 'Load estimated beta maps from training datasets ...'
    train_betas1_file = os.path.join(beta_dir, sid,
                            '%s_beta_train_s1_t%s.nii.gz'%(sid, test_run_idx))
    train_betas2_file = os.path.join(beta_dir, sid,
                            '%s_beta_train_s2_t%s.nii.gz'%(sid, test_run_idx))
    train_betas1 = nib.load(train_betas1_file).get_data()
    train_betas2 = nib.load(train_betas2_file).get_data()
    train_betas = np.concatenate((train_betas1, train_betas2), axis=3)
    print 'Load estimated beta maps from testing datasets ...'
    test_betas1_file = os.path.join(beta_dir, sid,
                            '%s_beta_val_s1_t%s.nii.gz'%(sid, test_run_idx))
    test_betas2_file = os.path.join(beta_dir, sid,
                            '%s_beta_val_s2_t%s.nii.gz'%(sid, test_run_idx))
    test_betas1 = nib.load(test_betas1_file).get_data()
    test_betas2 = nib.load(test_betas2_file).get_data()
    test_betas = np.concatenate((test_betas1, test_betas2), axis=3)
    # data normalization
    m = np.mean(train_betas, axis=3, keepdims=True)
    s = np.std(train_betas, axis=3, keepdims=True)
    train_betas = (train_betas - m) / (s + 1e-5)
    m = np.mean(test_betas, axis=3, keepdims=True)
    s = np.std(test_betas, axis=3, keepdims=True)
    test_betas = (test_betas - m) / (s + 1e-5)

    #-- get stimuli label info
    print 'Load stimuli label info ...'
    stim_label_list = get_stimuli_label(root_dir, sid)
    test_label = np.concatenate((stim_label_list[test_run_idx-1],
                                 stim_label_list[5+test_run_idx-1]))
    stim_label_list.pop(test_run_idx-1)
    stim_label_list.pop(5+test_run_idx-2)
    train_label = np.concatenate(tuple(item for item in stim_label_list))

    #-- svm-based searchlight
    clf_results = [np.zeros((64, 64, 33, rand_num)),
                   np.zeros((64, 64, 33, rand_num)),
                   np.zeros((64, 64, 33, rand_num)),
                   np.zeros((64, 64, 33, rand_num))]
    # for loop for voxel-wise searchlight
    mask_coord = niroi.get_roi_coord(mask_data)
    ccount = 0
    for c in mask_coord:
        ccount += 1
        print ccount
        cube_roi = np.zeros((64, 64, 33))
        cube_roi = niroi.cube_roi(cube_roi, c[0], c[1], c[2], 2, 1)
        cube_coord = niroi.get_roi_coord(cube_roi)
        train_x = []
        test_x = []
        for t in range(train_betas.shape[3]):
            vtr = niroi.get_voxel_value(cube_coord, train_betas[..., t])
            train_x.append(vtr.tolist())
        for t in range(test_betas.shape[3]):
            vtr = niroi.get_voxel_value(cube_coord, test_betas[..., t])
            test_x.append(vtr.tolist())
        train_x = np.array(train_x)
        test_x = np.array(test_x)
        # randomize labels
        for randn in range(rand_num):
            shuffle_idx = range(train_label.shape[0])
            random.Random(randn).shuffle(shuffle_idx)
            rtrain_label = train_label[shuffle_idx]
            shuffle_idx = range(test_label.shape[0])
            random.Random(randn).shuffle(shuffle_idx)
            rtest_label = test_label[shuffle_idx]
            # classifier
            # kernel can be specified as linear, poly, rbf, and sigmod
            kernel = 'rbf'
            clf = svm.SVC(kernel=kernel)
            clf.fit(train_x, rtrain_label)
            pred = clf.predict(test_x)
            for e in range(4):
                acc = np.sum(pred[rtest_label==(e+1)]==(e+1))*1.0/np.sum(rtest_label==(e+1))
                clf_results[e][c[0], c[1], c[2], randn] = acc
    
    # save to nifti
    for e in range(4):
        aff = nib.load(mask_file).affine
        result_file = os.path.join(work_dir, sid,
                            'rand_svm_t%s_e%s.nii.gz'%(test_run_idx, e+1))
        nibase.save2nifti(clf_results[e], aff, result_file)
        func2anat_mat = os.path.join(root_dir, 'workshop', 'glmmodel', 'nii',
                                     sid, 'ref_vol2highres.mat')
        t1brain_vol = os.path.join(root_dir, 'nii', sid+'P1', '3danat',
                                   'reg_fsl', 'T1_brain.nii.gz')
        if os.path.exists(func2anat_mat):
            result2_file = os.path.join(work_dir, sid,
                        'rand_svm_t%s_e%s_highres.nii.gz'%(test_run_idx, e+1))
            str_cmd = ['flirt', '-in', result_file, '-ref', t1brain_vol,
                       '-applyxfm', '-init', func2anat_mat, '-out',result2_file]
            os.system(' '.join(str_cmd))

def label2file(root_dir, sid):
    """Get subject's trial tag for each run."""
    beh_dir = os.path.join(root_dir, 'beh')
    # get subject name
    subj_name = {'S1': 'liqing', 'S2': 'zhangjipeng', 'S3': 'zhangdan',
                 'S4': 'wanghuicui', 'S5': 'zhuzhiyuan', 'S6': 'longhailiang',
                 'S7': 'liranran'}
    subj = subj_name[sid]
    # stimuli label list var
    for i in range(10):
        stim_label = []
        img_list = []
        # load experiment record
        record = os.path.join(beh_dir, 'trial_record_%s_run%s.csv'%(subj, i+1))
        record_info = open(record, 'r').readlines()
        record_info.pop(0)
        record_info = [line.strip().split(',') for line in record_info]
        for line in record_info:
            if not line[0] in img_list:
                img_list.append(line[0])
                stim_label.append(int(line[1]))
        outfile = '%s_stimuli_%s.csv'%(sid, i+1)
        with open(outfile, 'w') as f:
            for j in range(len(img_list)):
                f.write(','.join([img_list[j], str(stim_label[j])])+'\n')


if __name__=='__main__':
    root_dir = r'/nfs/diskstation/projects/emotionPro'

    # generate functional mask for each subject
    #gen_func_mask(root_dir, 'S1')

    #label2file(root_dir, 'S1')

    # SVM-based searchlight
    #svm_searchlight(root_dir, 'S1', 1)
    #svm_searchlight_cv(root_dir, 'S1')
    random_svm_searchlight(root_dir, 'S1', 1, 1000)
    #roi_svm(root_dir, 'S1', 'face_roi_mprm.nii.gz')

