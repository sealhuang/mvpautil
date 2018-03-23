# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import nibabel as nib
from sklearn import svm

from pynit.base import unpack as pyunpack
from nitools import roi as niroi
from nitools.roi import extract_mean_ts

def get_run_idx(scanlist_file, sid, par_idx):
    """Get run index from one subject's info based on par index."""
    [scan_info, subj_list] = pyunpack.readscanlist(scanlist_file)
    for subj in subj_list:
        if (subj.sess_ID[:2]==sid) and ('emo' in subj.run_info):
            [run_list, par_list] = subj.getruninfo('emo')
            if str(par_idx) in par_list:
                return subj.sess_ID, '00'+run_list[par_list.index(str(par_idx))]
    return None, None

def get_category_bold_ts(root_dir, subj, par_idx, roi):
    """Get mean BOLD time course for each emotion category."""
    doc_dir = os.path.join(root_dir, 'doc')
    nii_dir = os.path.join(root_dir, 'prepro')
    par_dir = os.path.join(root_dir, 'par', 'emo', 'emotion_wise')
    # read scanlist to get SID and run index
    scanlist_file = os.path.join(doc_dir, 'scanlist.csv')
    [sid, run_idx] = get_run_idx(scanlist_file, subj, par_idx)
    # get nii data
    nii_file = os.path.join(nii_dir,sid,run_idx, 'mni_sfunc_data_mcf_hp.nii.gz')
    nii_data = nib.load(nii_file).get_data()
    # read trial sequence for each emotion category
    par_file = os.path.join(par_dir, 'run_%s.par'%(par_idx))
    trial_info = open(par_file, 'r').readlines()
    trial_info = [line.strip().split('\t') for line in trial_info]
    # bold ts for each trial, code 1-Happy, 2-Fear, 3-Disgust, 4-Neutral
    trial_seq = {1: [], 2:[], 3:[], 4:[]}
    for line in trial_info:
        if int(line[1]):
            trial_seq[int(line[1])].append(int(float(line[0])/2))
    # data shape: trials x time x emotion
    bold_ts = np.zeros((22, 4, 4))
    # get BOLD time course for each trial
    roi_ts = niroi.extract_mean_ts(nii_data, roi)
    for i in range(4):
        for j in range(len(trial_seq[i+1])):
            bold_ts[j, :, i] = roi_ts[trial_seq[i+1][j]:(trial_seq[i+1][j]+4)]
    np.save('%s_run%s_bold_ts.npy'%(subj, par_idx), bold_ts)

def get_roi_mvps(subj, roi_coord):
    """Get features and the corresponding labels for training and testing
    data.
    """


def svm_searchlight(root_dir, subj, mask_file):
    """SVM based searchlight analysis."""
    # dir config
    doc_dir = os.path.join(root_dir, 'doc')
    nii_dir = os.path.join(root_dir, 'prepro')
    par_dir = os.path.join(root_dir, 'par', 'emo', 'emotion_wise')
    work_dir = os.path.join(root_dir, 'workshop', 'searchlight')
    # read mask file
    mask_file = os.path.join(work_dir, 'mask', 'func_mask.nii.gz')
    mask_data = nib.load(mask_file).get_data()
    mask_data = mask_data>0
    # svm results var
    clf_results = np.zeros((91, 109, 91, 6))
    # for loop for voxel-wise searchlight
    mask_coord = niroi.get_roi_coord(mask_data)
    for c in mask_coord:
        cube_roi = np.zeros((91, 109, 91))
        cube_roi = niroi.cube_roi(cube_roi, c[0], c[1], c[2], 2, 1)
        cube_coord = niroi.get_roi_coord(cube_roi)
        [x_train, y_train, x_test, y_test] = get_roi_mvps(subj, cube_coord)


def get_trial_sequence(root_dir, sid):
    """Get trial sequence for each emotion run."""
    beh_dir = os.path.join(root_dir, 'beh')
    info_dir = os.path.join(root_dir, 'workshop', 'trial_info')
    # get subject name
    subjs = {'S1': 'liqing', 'S2': 'zhangjipeng',
             'S3': 'zhangdan', 'S4': 'wanghuicui',
             'S5': 'zhuzhiyuan', 'S6': 'longhailiang',
             'S7': 'liranran'}
    subj = subjs[sid]
    # get run number for subject
    tag_list = os.listdir(beh_dir)
    tag_list = [line for line in tag_list if line[-3:]=='csv']
    run_num = len([line for line in tag_list if line.split('_')[2]==subj])
    # get trial information for each run
    for r in range(run_num):
        info_f = os.path.join(beh_dir, 'trial_tag_%s_run%s.csv'%(subj,r+1))
        info = open(info_f, 'r').readlines()
        info.pop(0)
        info = [line.strip().split(',') for line in info]
        # var init
        test_c = 0
        test_idx = [0] * len(info)
        img_names = []
        trial_tag = []
        rsp_tag = []
        for trial_idx in range(len(info)):
            line = info[trial_idx]
            if not line[0] in img_names:
                img_names.append(line[0])
            else:
                first_idx = img_names.index(line[0])
                test_c = test_c + 1
                test_idx[first_idx] = test_c
                test_idx[trial_idx] = test_c
                img_names.append(line[0])
            trial_tag.append(line[1])
            if line[2]=='NaN':
                rsp_tag.append('0')
            else:
                rsp_tag.append(line[2])
        # outfile
        outfile = os.path.join(info_dir, '%s_run%s.csv'%(sid, r+1))
        with open(outfile, 'w+') as f:
            f.write('trial,testid,emo_tag,resp_tag\n')
            for i in range(len(info)):
                f.write(','.join([str(i+1), str(test_idx[i]),
                                  trial_tag[i], rsp_tag[i]])+'\n')

def get_vxl_trial_rsp(root_dir):
    """Get multivoxel activity pattern for each srimulus
    from whole brain mask.
    """
    # directory config
    nii_dir = os.path.join(root_dir, 'prepro')
    rsp_dir = os.path.join(root_dir, 'workshop', 'trial_rsp', 'whole_brain')
    # load rois
    mask_data = nib.load(os.path.join(root_dir, 'group-level', 'rois',
                            'neurosynth', 'cube_rois_r2.nii.gz')).get_data()
    mask_data = mask_data>0
    # get scan info from scanlist
    scanlist_file = os.path.join(root_dir, 'doc', 'scanlist.csv')
    [scan_info, subj_list] = pyunpack.readscanlist(scanlist_file)

    for subj in subj_list:
        # get run infor for emo task
        sid = subj.sess_ID
        print sid
        subj_dir = os.path.join(nii_dir, sid)
        # get run index
        if not 'emo' in subj.run_info:
            continue
        [run_idx, par_idx] = subj.getruninfo('emo')
        # var for MVP
        for i in range(10):
            if not str(i+1) in par_idx:
                continue
            print 'Run %s'%(i+1)
            mvp_data = []
            # load cope data
            ipar = par_idx.index(str(i+1))
            run_dir = os.path.join(subj_dir, '00'+run_idx[ipar])
            print run_dir
            rsp_file = os.path.join(run_dir, 'mni_sfunc_data_mcf_hp.nii.gz')
            rsp = nib.load(rsp_file).get_data()
            # derive trial-wise response
            trsp = np.zeros((91, 109, 91, 88))
            for t in range(88):
                trsp[..., t] = (rsp[..., 4*t+5] + rsp[..., 4*t+6]) / 2
            # get MVP of mask
            vxl_coord = niroi.get_roi_coord(mask_data)
            for j in range(trsp.shape[3]):
                vtr = niroi.get_voxel_value(vxl_coord, trsp[..., j])
                mvp_data.append(vtr.tolist())
            outfile = os.path.join(rsp_dir, '%s_r%s_mvp.npy'%(sid[:2], i+1))
            np.save(outfile, np.array(mvp_data))

def emo_clf(root_dir, sid):
    """Emotion classification based on MVP."""
    # directory config
    tag_dir = os.path.join(root_dir, 'workshop', 'trial_info')
    rsp_dir = os.path.join(root_dir, 'workshop', 'trial_rsp', 'whole_brain')
    facc = {'1': [], '2': [], '3': [], '4': [], 'all': []}
    for fidx in range(10):
        train_mvp = None
        for i in range(10):
            # get MVP data
            mvp_file = os.path.join(rsp_dir, '%s_r%s_mvp.npy'%(sid, i+1))
            mvp = np.load(mvp_file)
            m = np.mean(mvp, axis=1, keepdims=True)
            s = np.std(mvp, axis=1, keepdims=True)
            mvp = (mvp - m) / (s + 1e-5)
            # get emotion tag
            tag_file = os.path.join(tag_dir, '%s_run%s.csv'%(sid, i+1))
            tag_info = open(tag_file).readlines()
            tag_info.pop(0)
            tag_info = [line.strip().split(',') for line in tag_info]
            tags = np.array([int(line[2]) for line in tag_info])
            if not i==fidx:
                if isinstance(train_mvp, np.ndarray):
                    train_mvp = np.concatenate((train_mvp, mvp), axis=0)
                    train_tag = np.concatenate((train_tag, tags), axis=0)
                else:
                    train_mvp = mvp
                    train_tag = tags
            else:
                test_mvp = mvp
                test_tag = tags
        # classification
        clf = svm.SVC(decision_function_shape='ovo', kernel='sigmoid')
        clf.fit(train_mvp, train_tag)
        pred = clf.predict(test_mvp)
        facc['all'].append(np.sum(pred==test_tag)*1.0 / pred.shape[0])
        for e in range(4):
            facc['%s'%(e+1)].append(np.sum(pred[test_tag==(e+1)]==test_tag[test_tag==(e+1)]) * 1.0 / test_tag[test_tag==(e+1)].shape[0])
    print 'Mean accuracy:'
    for k in facc:
        print k,
        print facc[k]
        print np.mean(facc[k])

def get_emo_ts(root_dir, seq):
    """Get neural activity time course of each roi on each emotion condition."""
    nii_dir = os.path.join(root_dir, 'nii')
    ppi_dir = os.path.join(root_dir, 'ppi')
    # load roi
    rois = nib.load(os.path.join(root_dir, 'group-level', 'rois', 'neurosynth',
                                 'cube_rois_r2.nii.gz')).get_data()
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
                    train_seq = [line[0] for line in seq[i+1]['train']
                                    if line[1]==(j+1)]
                    test_seq = [line[0] for line in seq[i+1]['test']
                                    if line[1]==(j+1)]
                    emo_data = np.zeros((91, 109, 91,
                                        len(train_seq)+len(test_seq)))
                    emo_data[..., :len(train_seq)] = train_cope[..., train_seq]
                    emo_data[..., len(train_seq):] = test_cope[..., test_seq]
                    # get time course for each roi
                    roi_ts = np.zeros((emo_data.shape[3], roi_num))
                    for k in range(roi_num):
                        roi_ts[:, k] = niroi.extract_mean_ts(emo_data,
                                                             rois==(k+1))
                    outfile = '%s_roi_ts_run%s_emo%s.npy'%(sid[:2], i+1, j+1)
                    outfile = os.path.join(ppi_dir, 'decovPPI', outfile)
                    np.save(outfile, roi_ts)

def get_trial_data(root_dir, seq):
    """Get neural activity time course of each roi on each emotion condition."""
    nii_dir = os.path.join(root_dir, 'nii')
    ppi_dir = os.path.join(root_dir, 'ppi')
    # load roi
    rois = nib.load(os.path.join(root_dir, 'group-level', 'rois', 'neurosynth',
                                 'cube_rois_r2.nii.gz')).get_data()
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
                # get time course for each roi
                train_x = np.zeros((train_cope.shape[3], roi_num))
                test_x = np.zeros((test_cope.shape[3], roi_num))
                for k in range(roi_num):
                    train_x[:, k]=niroi.extract_mean_ts(train_cope, rois==(k+1))
                    test_x[:, k] = niroi.extract_mean_ts(test_cope, rois==(k+1))
                train_y = [line[1] for line in seq[i+1]['train']]
                test_y = [line[1] for line in seq[i+1]['test']]
                # save dataset
                outfile = '%s_run%s_roi_data'%(sid[:2], i+1)
                outfile = os.path.join(ppi_dir, 'decovPPI', outfile)
                np.savez(outfile, train_x=train_x, train_y=train_y,
                                  test_x=test_x, test_y=test_y)

def get_trial_tag(root_dir, subj):
    """Get emotion tag for each trial"""
    beh_dir = os.path.join(root_dir, 'beh')
    par_dir = os.path.join(root_dir, 'par', 'emo')
    # get run number for subject
    tag_list = os.listdir(beh_dir)
    tag_list = [line for line in tag_list if line[-3:]=='csv']
    run_num = len([line for line in tag_list if line.split('_')[2]==subj])
    # sequence var
    tag_list = []
    for r in range(run_num):
        # dict for run `r+1`
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
            tag_list.append([img, emo])
        for test_idx in range(len(test_trials)):
            img = test_trials[test_idx][1].split('\\')[1]
            emo = int([line[1] for line in trial_info if line[0]==img][0])
            tag_list.append([img, emo])
    outfile = 'trial_tag.csv'
    f = open(outfile, 'w+')
    for item in tag_list:
        f.write(','.join([str(ele) for ele in item])+'\n')
    f.close()


if __name__=='__main__':
    root_dir = r'/nfs/diskstation/projects/emotionPro'

    ## get mean bold ts for each ROI
    #roi_file = os.path.join(root_dir, 'group-level', 'rois', 'neurosynth',
    #                        'cube_rois_r2.nii.gz')
    #roi_data = nib.load(roi_file).get_data()
    #get_category_bold_ts(root_dir, 'S1', 1, roi_data==20)

    # SVM-based searchlight
    svm_searchlight(root_dir, 'S1')

    #get_trial_sequence(root_dir, 'S1')
    #get_vxl_trial_rsp(root_dir)
    #emo_clf(root_dir, 'S1')
    
    #get_emo_ts(root_dir, seq)
    #get_conn(root_dir)
    #get_rand_conn(root_dir, 1000)
    #get_mvp_group_roi(root_dir)
    #get_trial_tag(root_dir, 'liqing')

