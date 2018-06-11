# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import nibabel as nib
from sklearn import svm

from pynit.base import unpack as pyunpack
from nitools import roi as niroi
from nitools import base as nibase
from nitools.roi import extract_mean_ts


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
    subj_info = {'S1': 'liqing', 'S2': 'zhangjipeng',
                 'S3': 'zhangdan', 'S4': 'wanghuicui',
                 'S5': 'zhuzhiyuan', 'S6': 'longhailiang',
                 'S7': 'liranran'}
    subj = subj_info[sid]
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

def svm_searchlight(root_dir, sid):
    """SVM based searchlight analysis."""
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
        clf = svm.SVC(kernel='sigmoid')
        clf.fit(train_x, train_label)
        pred = clf.predict(test_x)
        for e in range(4):
            acc = np.sum(pred[test_label==(e+1)]==(e+1))*1.0 / np.sum(test_label==(e+1))
            print acc
            clf_results[c[0], c[1], c[2], e] = acc
    # save to nifti
    aff = nib.load(mask_file).affine
    nibase.save2nifti(clf_results, aff,
                      os.path.join(work_dir,sid,'svm_acc.nii.gz'))

def random_svm_cope_searchlight(root_dir, subj):
    """SVM based searchlight analysis."""
    # dir config
    work_dir = os.path.join(root_dir, 'workshop', 'searchlight')
    # read mask file
    print 'Load mask data ...'
    mask_file = os.path.join(work_dir, 'mask', 'func_mask.nii.gz')
    mask_data = nib.load(mask_file).get_data()
    mask_data = mask_data>0
    # load nii data list
    print 'Load nii files ...'
    cope_list = get_subj_cope_list(root_dir, subj)
    # get trial sequence info
    print 'Load trial sequence info ...'
    tag_list = get_subj_cope_tag(root_dir, subj)
    for i in range(100):
        # svm results var
        clf_results = np.zeros((91, 109, 91, 4))
        # for loop for voxel-wise searchlight
        mask_coord = niroi.get_roi_coord(mask_data)
        ccount = 0
        for c in mask_coord:
            ccount += 1
            print ccount
            cube_roi = np.zeros((91, 109, 91))
            cube_roi = niroi.cube_roi(cube_roi, c[0], c[1], c[2], 2, 1)
            cube_coord = niroi.get_roi_coord(cube_roi)
            [train_x, train_y, test_x, test_y] = get_roi_cope_mvps(cope_list,
                                                                   tag_list,
                                                                   cube_coord)
            clf = svm.SVC(kernel='sigmoid')
            train_y = np.random.permutation(train_y)
            clf.fit(train_x, train_y)
            pred = clf.predict(test_x)
            for e in range(4):
                acc = np.sum(pred[test_y==(e+1)]==(e+1))*1.0 / np.sum(test_y==(e+1))
                print acc
                clf_results[c[0], c[1], c[2], e] = acc
        # save to nifti
        fsl_dir = os.getenv('FSL_DIR')
        template_file = os.path.join(fsl_dir, 'data', 'standard',
                                     'MNI152_T1_2mm_brain.nii.gz')
        aff = nib.load(template_file).affine
        nibase.save2nifti(clf_results, aff,
         os.path.join(work_dir, 'random_'+subj+'_svm_acc_cope_%s.nii.gz'%(i+1)))


if __name__=='__main__':
    root_dir = r'/nfs/diskstation/projects/emotionPro'

    # generate functional mask for each subject
    #gen_func_mask(root_dir, 'S1')

    # SVM-based searchlight
    svm_searchlight(root_dir, 'S1')
    #random_svm_cope_searchlight(root_dir, 'S1')

    # network analysis
    #get_emo_ts(root_dir, seq)
    #get_conn(root_dir)
    #get_rand_conn(root_dir, 1000)
    #get_mvp_group_roi(root_dir)

