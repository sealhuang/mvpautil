# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import random
import nibabel as nib
from sklearn import svm
from multiprocessing import Array, Pool

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

def get_stimuli_label(root_dir, sid, mode='objective'):
    """Get subject's trial tag for each run.

    mode options: `objective`  ->  image label
                  `subjective` ->  subject's response, 0 indicates non-response
                                   trials
                  `correct`    ->  correct trial, 0 indicates wrong response
    """
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
                if mode=='objective':
                    stim_label.append(int(line[1]))
                elif mode=='subjective':
                    if line[2]=='NaN':
                        stim_label.append(0)
                    else:
                        stim_label.append(int(line[2]))
                elif mode=='correct':
                    if line[1]==line[2]:
                        stim_label.append(int(line[1]))
                    else:
                        stim_label.append(0)
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
    #m = np.mean(train_betas, axis=3, keepdims=True)
    #s = np.std(train_betas, axis=3, keepdims=True)
    #train_betas = (train_betas - m) / (s + 1e-5)
    #m = np.mean(test_betas, axis=3, keepdims=True)
    #s = np.std(test_betas, axis=3, keepdims=True)
    #test_betas = (test_betas - m) / (s + 1e-5)
    for i in range(8):
        tmp = train_betas[..., (i*80):(i*80+80)]
        m = np.mean(tmp, axis=3, keepdims=True)
        s = np.std(tmp, axis=3, keepdims=True)
        train_betas[..., (i*80):(i*80+80)] = (tmp - m) / (s + 1e-5)
    for i in range(2):
        tmp = test_betas[..., (i*80):(i*80+80)]
        m = np.mean(tmp, axis=3, keepdims=True)
        s = np.std(tmp, axis=3, keepdims=True)
        test_betas[..., (i*80):(i*80+80)] = (tmp - m) / (s + 1e-5)

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
    
    # calculate mean classification accuracy across folds
    work_dir = os.path.join(root_dir, 'workshop', 'glmmodel', 'searchlight')
    subj_dir = os.path.join(work_dir, sid)

    print 'Calculate mean classification accuracy across CVs ...'
    # get cv results
    cv_files = [os.path.join(subj_dir, 'svm_rbf_t%s.nii.gz'%(i+1))
                for i in range(5)]
    cv_hr_files = [os.path.join(subj_dir, 'svm_rbf_t%s_highres.nii.gz'%(i+1))
                   for i in range(5)]
    cv_mean_file = os.path.join(subj_dir, 'svm_rbf_tmean.nii.gz')
    cv_hr_mean_file = os.path.join(subj_dir, 'svm_rbf_tmean_highres.nii.gz')
    # calculate mean accuracy
    cmd_str = ' '.join(['fslmaths'] + [' -add '.join(cv_files)] + \
                       ['-div', '5', cv_mean_file])
    os.system(cmd_str)
    cmd_str = ' '.join(['fslmaths'] + [' -add '.join(cv_hr_files)] + \
                       ['-div', '5', cv_hr_mean_file])
    os.system(cmd_str)
    
    # highres acc file to mni space
    highres2mni_mat = os.path.join(root_dir, 'nii', sid+'P1', '3danat',
                                   'reg_fsl', 'highres2standard_2mm.mat')
    mni_vol = os.path.join(os.environ['FSL_DIR'], 'data', 'standard',
                           'MNI152_T1_2mm_brain.nii.gz')
    mni_acc_file = os.path.join(subj_dir, 'svm_rbf_tmean_mni_linear.nii.gz')
    str_cmd = ['flirt', '-in', cv_hr_mean_file, '-ref', mni_vol,
               '-applyxfm', '-init', highres2mni_mat, '-out', mni_acc_file]
    os.system(' '.join(str_cmd))

def initpool(clf_result_e1, clf_result_e2, clf_result_e3, clf_result_e4):
    """Initializer for process pool."""
    global clf_e1
    global clf_e2
    global clf_e3
    global clf_e4
    clf_e1 = clf_result_e1
    clf_e2 = clf_result_e2
    clf_e3 = clf_result_e3
    clf_e4 = clf_result_e4

def randfunc(x, y,z,rand_num, train_betas, test_betas, train_label, test_label):
    """Sugar for randomization."""
    cube_roi = np.zeros((64, 64, 33))
    cube_roi = niroi.cube_roi(cube_roi, x, y, z, 2, 1)
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
    # get voxel index
    vxl_loc = x*64*33*rand_num + y*33*rand_num + z*rand_num
    print x, y, z
    # for loop for randomize
    for randn in xrange(0, rand_num):
        # randomize labels
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
        # store result
        acc = np.sum(pred[rtest_label==1]==1)*1.0/np.sum(rtest_label==1)
        clf_e1[vxl_loc+randn] = acc
        acc = np.sum(pred[rtest_label==2]==2)*1.0/np.sum(rtest_label==2)
        clf_e2[vxl_loc+randn] = acc
        acc = np.sum(pred[rtest_label==3]==3)*1.0/np.sum(rtest_label==3)
        clf_e3[vxl_loc+randn] = acc
        acc = np.sum(pred[rtest_label==4]==4)*1.0/np.sum(rtest_label==4)
        clf_e4[vxl_loc+randn] = acc

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
    clf_result_e1 = Array('d', np.zeros((64, 64, 33, rand_num)).flat)
    clf_result_e2 = Array('d', np.zeros((64, 64, 33, rand_num)).flat)
    clf_result_e3 = Array('d', np.zeros((64, 64, 33, rand_num)).flat)
    clf_result_e4 = Array('d', np.zeros((64, 64, 33, rand_num)).flat)
    # get mask
    mask_coord = niroi.get_roi_coord(mask_data)
    # randomize
    # setting up pool
    pool = Pool(initializer=initpool, initargs=(clf_result_e1, clf_result_e2,
                                                clf_result_e3, clf_result_e4),
                processes=20)
    p = [pool.apply_async(randfunc, args=(c[0], c[1], c[2], rand_num,
                                          train_betas, test_betas,
                                          train_label, test_label))
                          for c in mask_coord]
    pool.close()
    pool.join()
 
    #ccount = 0
    #for c in mask_coord:
    #    ccount += 1
    #    print ccount
    #    cube_roi = np.zeros((64, 64, 33))
    #    cube_roi = niroi.cube_roi(cube_roi, c[0], c[1], c[2], 2, 1)
    #    cube_coord = niroi.get_roi_coord(cube_roi)
    #    train_x = []
    #    test_x = []
    #    for t in range(train_betas.shape[3]):
    #        vtr = niroi.get_voxel_value(cube_coord, train_betas[..., t])
    #        train_x.append(vtr.tolist())
    #    for t in range(test_betas.shape[3]):
    #        vtr = niroi.get_voxel_value(cube_coord, test_betas[..., t])
    #        test_x.append(vtr.tolist())
    #    train_x = np.array(train_x)
    #    test_x = np.array(test_x)
    #    # randomize
    #    # setting up pool
    #    vxl_loc = c[0]*64*33*rand_num + c[1]*33*rand_num + c[2]*rand_num
    #    pool = Pool(initializer=initpool, initargs=(clf_result_e1,
    #                                                clf_result_e2,
    #                                                clf_result_e3,
    #                                                clf_result_e4),
    #                processes=20)
    #    p = [pool.apply_async(randfunc, args=(randn, train_label, test_label,
    #                                          train_x, test_x, vxl_loc))
    #                          for randn in xrange(0, rand_num)]
    #    pool.close()
    #    pool.join()
        #for rand in xrange(rand_num):
        #    # randomize labels
        #    shuffle_idx = range(train_label.shape[0])
        #    random.Random(randn).shuffle(shuffle_idx)
        #    rtrain_label = train_label[shuffle_idx]
        #    shuffle_idx = range(test_label.shape[0])
        #    random.Random(randn).shuffle(shuffle_idx)
        #    rtest_label = test_label[shuffle_idx]
        #    # classifier
        #    # kernel can be specified as linear, poly, rbf, and sigmod
        #    kernel = 'rbf'
        #    clf = svm.SVC(kernel=kernel)
        #    clf.fit(train_x, rtrain_label)
        #    pred = clf.predict(test_x)
        #    for e in range(4):
        #        acc = np.sum(pred[rtest_label==(e+1)]==(e+1))*1.0/np.sum(rtest_label==(e+1))
        #        clf_results[e][c[0], c[1], c[2], randn] = acc
    
    clf_results = [np.reshape(np.frombuffer(clf_result_e1.get_obj()),
                              (64, 64, 33, rand_num)),
                   np.reshape(np.frombuffer(clf_result_e2.get_obj()),
                              (64, 64, 33, rand_num)),
                   np.reshape(np.frombuffer(clf_result_e3.get_obj()),
                              (64, 64, 33, rand_num)),
                   np.reshape(np.frombuffer(clf_result_e4.get_obj()),
                              (64, 64, 33, rand_num))]

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

def stimseq2file(root_dir, sid):
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
        outfile = '%s_stimuli_%s.txt'%(sid, i+1)
        with open(outfile, 'w') as f:
            for j in range(len(img_list)):
                f.write(' '.join([img_list[j], str(stim_label[j])])+'\n')

def get_searchlight_p(root_dir, sid):
    """Get p value in SVM based searchlight analysis."""
    work_dir = os.path.join(root_dir, 'workshop', 'glmmodel', 'searchlight')
    subj_dir = os.path.join(work_dir, sid)

    cv_mean_file = os.path.join(subj_dir, 'svm_rbf_tmean.nii.gz')

    print 'Calculate mean classification accuracy across radnom CVs ...'
    # calculate mean accuracy for each emotion condition
    for e in range(4):
        cv_files = [os.path.join(subj_dir, 'rand_svm_t%s_e%s.nii.gz'%(i+1, e+1))
                    for i in range(5)]
        cv_mean_file = os.path.join(subj_dir, 'rand_svm_tmean_e%s.nii.gz'%(e+1))
        # calculate mean accuracy
        cmd_str = ' '.join(['fslmaths'] + [' -add '.join(cv_files)] + \
                           ['-div', '5', cv_mean_file])
        os.system(cmd_str)

    print 'Calculate p value for classification for each emotion condition ...'
    # read mask file
    print 'Load mask data ...'
    mask_file = os.path.join(work_dir, sid, 'func_mask.nii.gz')
    mask_data = nib.load(mask_file).get_data()
    mask_data = mask_data>0
    
    # load mean classification accuracy
    print 'Load mean classification accuracy data ...'
    mean_acc_img = nib.load(os.path.join(subj_dir, 'svm_rbf_tmean.nii.gz'))
    mean_acc = mean_acc_img.get_data()
    p_val = np.ones_like(mean_acc)
    for e in range(4):
        print 'Emotion %s'%(e+1)
        # p-value init
        rand_acc_file = os.path.join(subj_dir,'rand_svm_tmean_e%s.nii.gz'%(e+1))
        rand_acc = nib.load(rand_acc_file).get_data()
        # for loop for voxel-wise analysis
        mask_coord = niroi.get_roi_coord(mask_data)
        ccount = 0
        for c in mask_coord:
            ccount += 1
            print ccount
            rand_a = rand_acc[c[0], c[1], c[2]].copy()
            rand_a.sort()
            a = mean_acc[c[0], c[1], c[2], e]
            p_val[c[0], c[1], c[2], e] = np.sum(rand_a>a)*1.0/1000

    # save to nifti
    aff = nib.load(mask_file).affine
    p_file = os.path.join(subj_dir, 'svm_rbf_tmean_p.nii.gz')
    nibase.save2nifti(p_val, aff, p_file)
    func2anat_mat = os.path.join(root_dir, 'workshop', 'glmmodel', 'nii',
                                 sid, 'ref_vol2highres.mat')
    t1brain_vol = os.path.join(root_dir, 'nii', sid+'P1', '3danat',
                               'reg_fsl', 'T1_brain.nii.gz')
    if os.path.exists(func2anat_mat):
        p_hr_file = os.path.join(subj_dir, 'svm_rbf_tmean_p_highres.nii.gz')
        str_cmd = ['flirt', '-in', p_file, '-ref', t1brain_vol,
                   '-applyxfm', '-init', func2anat_mat, '-out', p_hr_file]
        os.system(' '.join(str_cmd))
        # get inverse p map
        ip_hr_file = os.path.join(subj_dir, 'svm_rbf_tmean_invp_highres.nii.gz')
        str_cmd = ['fslmaths', p_hr_file, '-mul', '-1', '-add', '1',
                   ip_hr_file]
        os.system(' '.join(str_cmd))

def fdr(root_dir, sid, alpha=0.05):
    """Conduct a FDR correction for p in SVM based searchlight analysis."""
    work_dir = os.path.join(root_dir, 'workshop', 'glmmodel', 'searchlight')

    print 'Load mask data ...'
    mask_file = os.path.join(work_dir, sid, 'func_mask.nii.gz')
    mask_data = nib.load(mask_file).get_data()
    mask_data = mask_data>0

    print 'Load CV-p data ...'
    p_file = os.path.join(work_dir, sid, 'svm_rbf_tmean_p.nii.gz')
    p_val = nib.load(p_file).get_data()

    fdr_thresh_file = os.path.join(work_dir, sid, 'fdr_threshold.csv')
    fdr_thresh_f = open(fdr_thresh_file, 'wb')

    # FDR calculation
    mask_coord = niroi.get_roi_coord(mask_data)
    vxl_num = mask_data.sum()
    fdr_p_val = np.zeros_like(p_val)
    # iter for each emotion type
    for e in range(fdr_p_val.shape[3]):
        p_vtr = np.zeros((vxl_num,))
        i = 0
        for c in mask_coord:
            a = p_val[c[0], c[1], c[2], e]
            p_vtr[i] = a
            i += 1
        p_vtr.sort()
        for i in range(vxl_num, 0, -1):
            a = p_vtr[i-1]
            if a <= (i*1.0/vxl_num*alpha):
                break
        print 'Threshold p for emotion %s: %s'%(e+1, a)
        thres = a
        fdr_p_val[p_val[..., e]<=thres, e] = 1
        fdr_thresh_f.write('Threshold p for emotion %s: %s\n'%(e+1, a))
    fdr_thresh_f.close()

    # save to nifti
    aff = nib.load(mask_file).affine
    fdr_file = os.path.join(work_dir, sid, 'svm_rbf_tmean_p_fdr.nii.gz')
    nibase.save2nifti(fdr_p_val, aff, fdr_file)
    func2anat_mat = os.path.join(root_dir, 'workshop', 'glmmodel', 'nii',
                                 sid, 'ref_vol2highres.mat')
    t1brain_vol = os.path.join(root_dir, 'nii', sid+'P1', '3danat',
                               'reg_fsl', 'T1_brain.nii.gz')
    if os.path.exists(func2anat_mat):
        fdr_hr_file =os.path.join(work_dir, sid,
                                  'svm_rbf_tmean_p_fdr_highres.nii.gz')
        str_cmd = ['flirt', '-in', fdr_file, '-ref', t1brain_vol,
                   '-applyxfm', '-init', func2anat_mat,
                   '-interp', 'nearestneighbour',
                   '-out', fdr_hr_file]
        os.system(' '.join(str_cmd))

def p2surf(root_dir, sid):
    """Register fdr-corrected p value from volume to surface."""
    subj_dir = os.path.join(root_dir, 'workshop', 'glmmodel', 'searchlight',sid)
    work_dir = os.path.join(subj_dir, 'surf_reg')
    if not os.path.exists(work_dir):
        os.system('mkdir %s'%(work_dir))

    # set subject dir
    os.environ['SUBJECTS_DIR'] = os.path.join(root_dir, 'freesurfer')

    # get fdr masked p file
    fdr_p_file = os.path.join(work_dir, 'svm_rbf_tmean_p_masked.nii.gz')
    p_file = os.path.join(subj_dir, 'svm_rbf_tmean_p.nii.gz')
    fdr_mask_file = os.path.join(subj_dir, 'svm_rbf_tmean_p_fdr.nii.gz')
    str_cmd = ['fslmaths', p_file, '-mul -1', '-add 1',
               '-mul', fdr_mask_file, fdr_p_file]
    os.system(' '.join(str_cmd))

    # split p file
    str_cmd = ['fslsplit', fdr_p_file, 'emo_p_']
    os.system(' '.join(str_cmd))
    str_cmd = ['mv', 'emo_p_*', work_dir]
    os.system(' '.join(str_cmd))

    # register volume to surface
    reg_lta = os.path.join(root_dir, 'workshop', 'glmmodel', 'nii', sid,
                           'surf_reg', 'register.lta')
    for i in range(4):
        vol_file = os.path.join(work_dir, 'emo_p_000%s.nii.gz'%(i))
        for h in ['lh', 'rh']:
            surf_file = os.path.join(work_dir, '%s_fdr_p_emo_%s.mgh'%(h, i))
            str_cmd = ['mri_vol2surf', '--mov', vol_file, '--reg', reg_lta,
                       '--projfrac', '0.5', '--interp', 'nearest',
                       '--hemi', h, '--o', surf_file]
            os.system(' '.join(str_cmd))

def roi_clf(root_dir, sid):
    """Get classification accuracy for each emotion-related ROI."""
    # dir config
    subj_dir = os.path.join(root_dir, 'workshop', 'glmmodel', 'conn', sid)
    
    # load mask file
    mask_file = os.path.join(root_dir, 'group-level', 'rois', 'power264',
                             'sel_emotion_rois.nii.gz')
    mask = nib.load(mask_file).get_data()
    roi_num = int(mask.max())
    acc_mtx = np.zeros((5, roi_num, 6))

    # calculate classification accuracy
    for r in range(1, 6):
        # load estimated beta maps
        print 'Load estimated beta maps from training datasets ...'
        train_beta1_file = os.path.join(subj_dir,
                                '%s_beta_train_s1_t%s_mni.nii.gz'%(sid, r))
        train_beta2_file = os.path.join(subj_dir,
                                '%s_beta_train_s2_t%s_mni.nii.gz'%(sid, r))
        train_beta1 = nib.load(train_beta1_file).get_data()
        train_beta2 = nib.load(train_beta2_file).get_data()
        train_beta = np.concatenate((train_beta1, train_beta2), axis=3)
        print 'Load estimated beta maps from testing datasets ...'
        test_beta1_file = os.path.join(subj_dir,
                                       '%s_beta_val_s1_t%s_mni.nii.gz'%(sid, r))
        test_beta2_file = os.path.join(subj_dir,
                                       '%s_beta_val_s2_t%s_mni.nii.gz'%(sid, r))
        test_beta1 = nib.load(test_beta1_file).get_data()
        test_beta2 = nib.load(test_beta2_file).get_data()
        test_beta = np.concatenate((test_beta1, test_beta2), axis=3)
        # data normalization
        for i in range(8):
            tmp = train_beta[..., (i*80):(i*80+80)]
            m = np.mean(tmp, axis=3, keepdims=True)
            s = np.std(tmp, axis=3, keepdims=True)
            train_beta[..., (i*80):(i*80+80)] = (tmp - m) / (s + 1e-5)
        for i in range(2):
            tmp = test_beta[..., (i*80):(i*80+80)]
            m = np.mean(tmp, axis=3, keepdims=True)
            s = np.std(tmp, axis=3, keepdims=True)
            test_beta[..., (i*80):(i*80+80)] = (tmp - m) / (s + 1e-5)
        print train_beta.shape
        print test_beta.shape
 
        # get stimuli label info
        print 'Load stimuli label info ...'
        stim_label_list = get_stimuli_label(root_dir, sid, mode='subjective')
        test_label = np.concatenate((stim_label_list[r-1],
                                     stim_label_list[5+r-1]))
        stim_label_list.pop(r-1)
        stim_label_list.pop(5+r-2)
        train_label = np.concatenate(tuple(item for item in stim_label_list))
        #print train_label.shape
        #print test_label.shape
        
        # for loop for roi-wise classification
        for c in range(roi_num):
            roi_idx = c + 1
            cube_coord = niroi.get_roi_coord(mask==roi_idx)
            
            e = 0
            for e1 in range(1, 5):
                for e2 in range(e1+1, 5):
                    print '%s VS. %s ...'%(e1, e2)
                    train_x = []
                    test_x = []
                    train_smp_idx = [t for t in range(train_beta.shape[3])
                                if (train_label[t]==e1 or train_label[t]==e2)]
                    test_smp_idx = [t for t in range(test_beta.shape[3])
                                    if (test_label[t]==e1 or test_label[t]==e2)]
                    for t in train_smp_idx:
                        vtr = niroi.get_voxel_value(cube_coord,
                                                    train_beta[..., t])
                        train_x.append(vtr.tolist())
                    for t in test_smp_idx:
                        vtr = niroi.get_voxel_value(cube_coord,
                                                    test_beta[..., t])
                        test_x.append(vtr.tolist())
                    train_x = np.array(train_x)
                    test_x = np.array(test_x)
                    train_y = train_label[train_smp_idx]
                    test_y = test_label[test_smp_idx]
                    train_y[train_y>e1] = 0
                    train_y[train_y>0] = 1
                    test_y = test_label[test_smp_idx]
                    test_y[test_y>e1] = 0
                    test_y[test_y>0] = 1
                    print train_x.shape
                    print test_x.shape
                    print train_y.shape
                    print test_y.shape
                    
                    # classifier
                    # kernel can be specified as linear, poly, rbf, and sigmod
                    kernel = 'rbf'
                    clf = svm.SVC(kernel=kernel)
                    clf.fit(train_x, train_y)
                    pred = clf.predict(test_x)
                    acc = np.sum(pred==test_y)*1.0 / test_y.shape[0]
                    print acc
                    acc_mtx[r-1, c, e] = acc
                    e = e + 1
                    print '---------------------'

            #train_x = []
            #test_x = []
            #for t in range(train_beta.shape[3]):
            #    vtr = niroi.get_voxel_value(cube_coord, train_beta[..., t])
            #    train_x.append(vtr.tolist())
            #for t in range(test_beta.shape[3]):
            #    vtr = niroi.get_voxel_value(cube_coord, test_beta[..., t])
            #    test_x.append(vtr.tolist())
            #train_x = np.array(train_x)
            #test_x = np.array(test_x)
            ## classifier
            ## kernel can be specified as linear, poly, rbf, and sigmod
            #kernel = 'rbf'
            #clf = svm.SVC(kernel=kernel)
            #clf.fit(train_x, train_label)
            #pred = clf.predict(test_x)
            #for e in range(4):
            #    acc = np.sum(pred[test_label==(e+1)]==(e+1))*1.0 / np.sum(test_label==(e+1))
            #    print acc
            #    acc_mtx[r-1, c, e] = acc
    
    # save data
    np.save('%s_roi_clf_acc.npy'%(sid), acc_mtx)

def roi_clf_csv(root_dir, sid):
    """Get classification accuracy for each emotion-related ROI."""
    # load roi info
    roi_info = open('sel_emotion_rois.csv').readlines()
    roi_info.pop(0)
    roi_info = [line.strip().split(',') for line in roi_info]

    # dir config
    subj_dir = os.path.join(root_dir, 'workshop', 'glmmodel', 'conn', sid)
    
    ## load mask file
    #mask_file = os.path.join(root_dir, 'group-level', 'rois', 'power264',
    #                         'sel_emotion_rois.nii.gz')
    #mask = nib.load(mask_file).get_data()
    #roi_num = int(mask.max())
    roi_num = len(roi_info)
    acc_mtx = np.zeros((5, roi_num, 6))

    # calculate classification accuracy
    for r in range(1, 6):
        # load estimated beta maps
        print 'Load estimated beta maps from training datasets ...'
        train_beta1_file = os.path.join(subj_dir,
                                '%s_beta_train_s1_t%s_mni.nii.gz'%(sid, r))
        train_beta2_file = os.path.join(subj_dir,
                                '%s_beta_train_s2_t%s_mni.nii.gz'%(sid, r))
        train_beta1 = nib.load(train_beta1_file).get_data()
        train_beta2 = nib.load(train_beta2_file).get_data()
        train_beta = np.concatenate((train_beta1, train_beta2), axis=3)
        print 'Load estimated beta maps from testing datasets ...'
        test_beta1_file = os.path.join(subj_dir,
                                       '%s_beta_val_s1_t%s_mni.nii.gz'%(sid, r))
        test_beta2_file = os.path.join(subj_dir,
                                       '%s_beta_val_s2_t%s_mni.nii.gz'%(sid, r))
        test_beta1 = nib.load(test_beta1_file).get_data()
        test_beta2 = nib.load(test_beta2_file).get_data()
        test_beta = np.concatenate((test_beta1, test_beta2), axis=3)
        # data normalization
        for i in range(8):
            tmp = train_beta[..., (i*80):(i*80+80)]
            m = np.mean(tmp, axis=3, keepdims=True)
            s = np.std(tmp, axis=3, keepdims=True)
            train_beta[..., (i*80):(i*80+80)] = (tmp - m) / (s + 1e-5)
        for i in range(2):
            tmp = test_beta[..., (i*80):(i*80+80)]
            m = np.mean(tmp, axis=3, keepdims=True)
            s = np.std(tmp, axis=3, keepdims=True)
            test_beta[..., (i*80):(i*80+80)] = (tmp - m) / (s + 1e-5)
        print train_beta.shape
        print test_beta.shape
 
        # get stimuli label info
        print 'Load stimuli label info ...'
        stim_label_list = get_stimuli_label(root_dir, sid, mode='subjective')
        test_label = np.concatenate((stim_label_list[r-1],
                                     stim_label_list[5+r-1]))
        stim_label_list.pop(r-1)
        stim_label_list.pop(5+r-2)
        train_label = np.concatenate(tuple(item for item in stim_label_list))
        #print train_label.shape
        #print test_label.shape
        
        # for loop for roi-wise classification
        for c in range(roi_num):
            line = roi_info[c]
            i = int((90.0 - float(line[3])) / 2)
            j = int((float(line[4]) + 126) / 2)
            k = int((float(line[5]) + 72) / 2)
            mask = niroi.sphere_roi(np.zeros((91, 109, 91)), i, j, k, 3, 1)
            cube_coord = niroi.get_roi_coord(mask==1)

            #roi_idx = c + 1
            #cube_coord = niroi.get_roi_coord(mask==roi_idx)
            
            e = 0
            for e1 in range(1, 5):
                for e2 in range(e1+1, 5):
                    print '%s VS. %s ...'%(e1, e2)
                    train_x = []
                    test_x = []
                    train_smp_idx = [t for t in range(train_beta.shape[3])
                                if (train_label[t]==e1 or train_label[t]==e2)]
                    test_smp_idx = [t for t in range(test_beta.shape[3])
                                    if (test_label[t]==e1 or test_label[t]==e2)]
                    for t in train_smp_idx:
                        vtr = niroi.get_voxel_value(cube_coord,
                                                    train_beta[..., t])
                        train_x.append(vtr.tolist())
                    for t in test_smp_idx:
                        vtr = niroi.get_voxel_value(cube_coord,
                                                    test_beta[..., t])
                        test_x.append(vtr.tolist())
                    train_x = np.array(train_x)
                    test_x = np.array(test_x)
                    train_y = train_label[train_smp_idx]
                    test_y = test_label[test_smp_idx]
                    train_y[train_y>e1] = 0
                    train_y[train_y>0] = 1
                    test_y = test_label[test_smp_idx]
                    test_y[test_y>e1] = 0
                    test_y[test_y>0] = 1
                    print train_x.shape
                    print test_x.shape
                    print train_y.shape
                    print test_y.shape
                    
                    # classifier
                    # kernel can be specified as linear, poly, rbf, and sigmod
                    kernel = 'rbf'
                    clf = svm.SVC(kernel=kernel)
                    clf.fit(train_x, train_y)
                    pred = clf.predict(test_x)
                    acc = np.sum(pred==test_y)*1.0 / test_y.shape[0]
                    print acc
                    acc_mtx[r-1, c, e] = acc
                    e = e + 1
                    print '---------------------'

            #train_x = []
            #test_x = []
            #for t in range(train_beta.shape[3]):
            #    vtr = niroi.get_voxel_value(cube_coord, train_beta[..., t])
            #    train_x.append(vtr.tolist())
            #for t in range(test_beta.shape[3]):
            #    vtr = niroi.get_voxel_value(cube_coord, test_beta[..., t])
            #    test_x.append(vtr.tolist())
            #train_x = np.array(train_x)
            #test_x = np.array(test_x)
            ## classifier
            ## kernel can be specified as linear, poly, rbf, and sigmod
            #kernel = 'rbf'
            #clf = svm.SVC(kernel=kernel)
            #clf.fit(train_x, train_label)
            #pred = clf.predict(test_x)
            #for e in range(4):
            #    acc = np.sum(pred[test_label==(e+1)]==(e+1))*1.0 / np.sum(test_label==(e+1))
            #    print acc
            #    acc_mtx[r-1, c, e] = acc
    
    # save data
    np.save('%s_roi_clf_acc_sphere3.npy'%(sid), acc_mtx)

def random_roi_clf(root_dir, sid, test_run_idx, rand_num):
    """Generate a NULL distribution for roi-based roi analysis."""
    print 'ROI analysis on Subject %s - test run %s'%(sid, test_run_idx)
    # dir config
    subj_dir = os.path.join(root_dir, 'workshop', 'glmmodel', 'conn', sid)
    
    # read mask file
    print 'Load mask data ...'
    mask_file = os.path.join(root_dir, 'group-level', 'rois', 'power264',
                             'sel_emotion_rois.nii.gz')
    mask = nib.load(mask_file).get_data()
    roi_num = int(mask.max())
    acc_mtx = np.zeros((rand_num, roi_num, 6))

    # load estimated beta maps
    print 'Load estimated beta maps from training datasets ...'
    train_beta1_file = os.path.join(subj_dir,
                        '%s_beta_train_s1_t%s_mni.nii.gz'%(sid, test_run_idx))
    train_beta2_file = os.path.join(subj_dir,
                        '%s_beta_train_s2_t%s_mni.nii.gz'%(sid, test_run_idx))
    train_beta1 = nib.load(train_beta1_file).get_data()
    train_beta2 = nib.load(train_beta2_file).get_data()
    train_beta = np.concatenate((train_beta1, train_beta2), axis=3)
    print 'Load estimated beta maps from testing datasets ...'
    test_beta1_file = os.path.join(subj_dir,
                        '%s_beta_val_s1_t%s_mni.nii.gz'%(sid, test_run_idx))
    test_beta2_file = os.path.join(subj_dir,
                        '%s_beta_val_s2_t%s_mni.nii.gz'%(sid, test_run_idx))
    test_beta1 = nib.load(test_beta1_file).get_data()
    test_beta2 = nib.load(test_beta2_file).get_data()
    test_beta = np.concatenate((test_beta1, test_beta2), axis=3)
    # data normalization
    for i in range(8):
        tmp = train_beta[..., (i*80):(i*80+80)]
        m = np.mean(tmp, axis=3, keepdims=True)
        s = np.std(tmp, axis=3, keepdims=True)
        train_beta[..., (i*80):(i*80+80)] = (tmp - m) / (s + 1e-5)
    for i in range(2):
        tmp = test_beta[..., (i*80):(i*80+80)]
        m = np.mean(tmp, axis=3, keepdims=True)
        s = np.std(tmp, axis=3, keepdims=True)
        test_beta[..., (i*80):(i*80+80)] = (tmp - m) / (s + 1e-5)
    print train_beta.shape
    print test_beta.shape
 
    # get stimuli label info
    print 'Load stimuli label info ...'
    stim_label_list = get_stimuli_label(root_dir, sid)
    test_label = np.concatenate((stim_label_list[test_run_idx-1],
                                 stim_label_list[5+test_run_idx-1]))
    stim_label_list.pop(test_run_idx-1)
    stim_label_list.pop(5+test_run_idx-2)
    train_label = np.concatenate(tuple(item for item in stim_label_list))

    # for loop for randomization
    for r in range(rand_num):
        rand_train_label = train_label[np.random.permutation(train_label.shape[0])]
        rand_test_label = test_label[np.random.permutation(test_label.shape[0])]

        # for loop for roi-wise classification
        for c in range(roi_num):
            roi_idx = c + 1
            cube_coord = niroi.get_roi_coord(mask==roi_idx)
            
            e = 0
            for e1 in range(1, 5):
                for e2 in range(e1+1, 5):
                    print '%s VS. %s ...'%(e1, e2)
                    train_x = []
                    test_x = []
                    train_smp_idx = [t for t in range(train_beta.shape[3])
                        if (rand_train_label[t]==e1 or rand_train_label[t]==e2)]
                    test_smp_idx = [t for t in range(test_beta.shape[3])
                        if (rand_test_label[t]==e1 or rand_test_label[t]==e2)]
                    for t in train_smp_idx:
                        vtr = niroi.get_voxel_value(cube_coord,
                                                    train_beta[..., t])
                        train_x.append(vtr.tolist())
                    for t in test_smp_idx:
                        vtr = niroi.get_voxel_value(cube_coord,
                                                    test_beta[..., t])
                        test_x.append(vtr.tolist())
                    train_x = np.array(train_x)
                    test_x = np.array(test_x)
                    train_y = rand_train_label[train_smp_idx]
                    test_y = rand_test_label[test_smp_idx]
                    train_y[train_y>e1] = 0
                    train_y[train_y>0] = 1
                    test_y = rand_test_label[test_smp_idx]
                    test_y[test_y>e1] = 0
                    test_y[test_y>0] = 1
                    print train_x.shape
                    print test_x.shape
                    print train_y.shape
                    print test_y.shape
                    
                    # classifier
                    # kernel can be specified as linear, poly, rbf, and sigmod
                    kernel = 'rbf'
                    clf = svm.SVC(kernel=kernel)
                    clf.fit(train_x, train_y)
                    pred = clf.predict(test_x)
                    acc = np.sum(pred==test_y)*1.0 / test_y.shape[0]
                    print acc
                    acc_mtx[r, c, e] = acc
                    e = e + 1
                    print '---------------------'
    # save data
    np.save('%s_roi_clf_rand_acc_t%s.npy'%(sid, test_run_idx), acc_mtx)


if __name__=='__main__':
    root_dir = r'/nfs/diskstation/projects/emotionPro'

    # generate functional mask for each subject
    #gen_func_mask(root_dir, 'S1')

    #stimseq2file(root_dir, 'S1')

    # SVM-based searchlight
    #svm_searchlight(root_dir, 'S1', 1)
    #svm_searchlight_cv(root_dir, 'S1')
    #random_svm_searchlight(root_dir, 'S1', 1000, 10)
    #get_searchlight_p(root_dir, 'S1')
    #fdr(root_dir, 'S1', alpha=0.05)
    #p2surf(root_dir, 'S1')

    #roi_svm(root_dir, 'S1', 'face_roi_mprm.nii.gz')
    #roi_clf(root_dir, 'S1')
    roi_clf_csv(root_dir, 'S1')
    #random_roi_clf(root_dir, 'S1', 1, 1000)

