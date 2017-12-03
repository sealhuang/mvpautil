# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import nibabel as nib
from scipy import io as sio

from pynit.base import unpack as pyunpack
from nitools import roi as niroi
from nitools.roi import extract_mean_ts


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
        train_trial_file = os.path.join(par_dir, 'trial_seq_%s_train.txt'%(r+1))
        test_trial_file = os.path.join(par_dir, 'trial_seq_%s_test.txt'%(r+1))
        if not os.path.exists(train_trial_file):
            print '%s does not exists'%(train_trial_file)
            continue
        # dict for run `r+1`
        seq[r+1] = {'train': [], 'test': []}
        train_trials = open(train_trial_file, 'r').readlines()
        test_trials = open(test_trial_file, 'r').readlines()
        train_trials = [line.strip().split(',') for line in train_trials]
        test_trials = [line.strip().split(',') for line in test_trials]
        trial_tag_f = os.path.join(beh_dir, 'trial_tag_%s_run%s.csv'%(subj,r+1))
        trial_tag = open(trial_tag_f, 'r').readlines()
        trial_tag.pop(0)
        trial_tag = [line.strip().split(',') for line in trial_tag]
        for train_idx in range(len(train_trials)):
            img = train_trials[train_idx][1].split('\\')[1]
            emo = int([line[1] for line in trial_tag if line[0]==img][0])
            subj_emo = [line[2] for line in trial_tag if line[0]==img][0]
            if subj_emo=='NaN':
                subj_emo = 0
            else:
                subj_emo = int(subj_emo)
            seq[r+1]['train'].append([train_idx, emo, subj_emo])
        for test_idx in range(len(test_trials)):
            img = test_trials[test_idx][1].split('\\')[1]
            emo = int([line[1] for line in trial_tag if line[0]==img][0])
            subj_emo = [line[2] for line in trial_tag if line[0]==img][0]
            if subj_emo=='NaN':
                subj_emo = 0
            else:
                subj_emo = int(subj_emo)
            seq[r+1]['test'].append([test_idx, emo, subj_emo])
    return seq

#def get_emo_sequence(root_dir, subj):
#    """Get trial sequence for each emotion condition."""
#    beh_dir = os.path.join(root_dir, 'beh')
#    par_dir = os.path.join(root_dir, 'par', 'emo')
#    # get run number for subject
#    tag_list = os.listdir(beh_dir)
#    tag_list = [line for line in tag_list if line[-3:]=='csv']
#    run_num = len([line for line in tag_list if line.split('_')[2]==subj])
#    # sequence var
#    seq = {}
#    for r in range(run_num):
#        # dict for run `r+1`
#        seq[r+1] = {}
#        train_trial_file = os.path.join(par_dir, 'trial_seq_%s_train.txt'%(r+1))
#        test_trial_file = os.path.join(par_dir, 'trial_seq_%s_test.txt'%(r+1))
#        train_trials = open(train_trial_file, 'r').readlines()
#        test_trials = open(test_trial_file, 'r').readlines()
#        train_trials = [line.strip().split(',') for line in train_trials]
#        test_trials = [line.strip().split(',') for line in test_trials]
#        trial_info_f = os.path.join(beh_dir,'trial_tag_%s_run%s.csv'%(subj,r+1))
#        trial_info = open(trial_info_f, 'r').readlines()
#        trial_info.pop(0)
#        trial_info = [line.strip().split(',') for line in trial_info]
#        for train_idx in range(len(train_trials)):
#            img = train_trials[train_idx][1].split('\\')[1]
#            emo = int([line[1] for line in trial_info if line[0]==img][0])
#            if not emo in seq[r+1]:
#                seq[r+1][emo] = {'train': [train_idx]}
#            else:
#                seq[r+1][emo]['train'].append(train_idx)
#        for test_idx in range(len(test_trials)):
#            img = test_trials[test_idx][1].split('\\')[1]
#            emo = int([line[1] for line in trial_info if line[0]==img][0])
#            if not 'test' in seq[r+1][emo]:
#                seq[r+1][emo]['test'] = [test_idx]
#            else:
#                seq[r+1][emo]['test'].append(test_idx)
#    return seq
#
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

def get_conn(root_dir):
    """Get connectivity matrix."""
    ppi_dir = os.path.join(root_dir, 'ppi', 'decovPPI')
    conn_dict = {}
    for i in range(7):
        #roi_idx = range(0, 10) + range(11, 49) + range(50, 75) + range(76, 86) + [89, 90] + range(92, 111) + range(112, 166) + range(167, 176) + range(177, 191)
        roi_idx = range(37)
        print 'ROI number: %s'%(len(roi_idx))
        conn_dict['s%s'%(i+1)] = np.zeros((len(roi_idx), len(roi_idx), 4))
        #conn_dict['s%s'%(i+1)] = np.zeros((41, 41, 4))
        for j in range(4):
            ts = None
            for k in range(10):
                ts_name = r'S%s_roi_ts_run%s_emo%s.npy'%(i+1, k+1, j+1)
                ts_file = os.path.join(ppi_dir, 'roi_ts','rois_meta_r2',ts_name)
                if not os.path.exists(ts_file):
                    print '%s not exists'%(ts_name)
                    continue
                tmp = np.load(ts_file)
                if isinstance(ts, np.ndarray):
                    tmp = tmp[:, roi_idx]
                    ts = np.concatenate((ts, tmp), axis=0)
                else:
                    ts = tmp[:, roi_idx]
            print ts.shape
            conn_dict['s%s'%(i+1)][..., j] = np.corrcoef(ts.T)
        outname = r's%s_conn.npy'%(i+1)
        np.save(os.path.join(ppi_dir, outname), conn_dict['s%s'%(i+1)])
    outfile = os.path.join(ppi_dir, 'conn_mtx.mat')
    sio.savemat(outfile, conn_dict)

def get_rand_conn(root_dir, rand_num):
    """Get connectivity matrix."""
    ppi_dir = os.path.join(root_dir, 'ppi', 'decovPPI')
    conn_dict = {}
    for i in range(7):
        conn_dict['s%s'%(i+1)] = np.zeros((37, 37, 4, rand_num))
        ts = None
        for j in range(10):
            for k in range(4):
                ts_name = r'S%s_roi_ts_run%s_emo%s.npy'%(i+1, j+1, k+1)
                ts_file = os.path.join(ppi_dir, 'roi_ts','rois_meta', ts_name)
                if not os.path.exists(ts_file):
                    print '%s not exists'%(ts_name)
                    continue
                tmp = np.load(ts_file)
                if isinstance(ts, np.ndarray):
                    ts = np.concatenate((ts, tmp), axis=0)
                else:
                    ts = tmp
        print ts.shape
        for r in range(rand_num):
            permutated_idx  = np.random.permutation(ts.shape[0])
            parts = ts.shape[0] / 4
            for c in range(4):
                tmp = ts[permutated_idx[(c*parts):(c*parts+parts)], :]
                conn_dict['s%s'%(i+1)][..., c, r] = np.corrcoef(tmp.T)
        outname = r's%s_rand_conn.npy'%(i+1)
        np.save(os.path.join(ppi_dir, outname), conn_dict['s%s'%(i+1)])
    outfile = os.path.join(ppi_dir, 'rand_conn_mtx.mat')
    sio.savemat(outfile, conn_dict)

def get_mvp_group_roi(root_dir):
    """Get multivoxel activity pattern for each srimulus from each ROI."""
    # directory config
    nii_dir = os.path.join(root_dir, 'nii')
    ppi_dir = os.path.join(root_dir, 'ppi')
    # load rois
    #mask_data = nib.load(os.path.join(ppi_dir, 'cube_rois.nii.gz')).get_data()
    mask_data = nib.load(os.path.join(root_dir, 'group-level', 'rois',
                                'neurosynth', 'cube_rois.nii.gz')).get_data()
    roi_num = int(mask_data.max())
    # get scan info from scanlist
    scanlist_file = os.path.join(root_dir, 'doc', 'scanlist.csv')
    [scan_info, subj_list] = pyunpack.readscanlist(scanlist_file)

    for subj in subj_list:
        # get run infor for emo task
        sid = subj.sess_ID
        subj_dir = os.path.join(nii_dir, sid, 'emo')
        # get run index
        if not 'emo' in subj.run_info:
            continue
        [run_idx, par_idx] = subj.getruninfo('emo')
        # var for MVP
        mvp_dict = {}
        for r in range(roi_num):
            mvp_dict['roi_%s'%(r+1)] = []
        for i in range(10):
            if str(i+1) in par_idx:
                print 'Run %s'%(i+1)
                # load cope data
                ipar = par_idx.index(str(i+1))
                run_dir = os.path.join(subj_dir, '00'+run_idx[ipar])
                print run_dir
                trn_file = os.path.join(run_dir, 'train_merged_cope.nii.gz')
                test_file = os.path.join(run_dir, 'test_merged_cope.nii.gz')
                trn_cope = nib.load(trn_file).get_data()
                test_cope = nib.load(test_file).get_data()
                run_cope = np.concatenate((trn_cope, test_cope), axis=3)
                # XXX: remove mean cope from each trial
                mean_cope = np.mean(run_cope, axis=3, keepdims=True)
                run_cope = run_cope - mean_cope
                # get MVP for each ROI
                for r in range(roi_num):
                    roi_mask = mask_data.copy()
                    roi_mask[roi_mask!=(r+1)] = 0
                    roi_mask[roi_mask==(r+1)] = 1
                    roi_coord = niroi.get_roi_coord(roi_mask)
                    for j in range(run_cope.shape[3]):
                        vtr = niroi.get_voxel_value(roi_coord, run_cope[..., j])
                        mvp_dict['roi_%s'%(r+1)].append(vtr.tolist())
        for roi in mvp_dict:
            mvp_dict[roi] = np.array(mvp_dict[roi])
        outfile = r'%s_roi_mvp.mat'%(sid)
        sio.savemat(outfile, mvp_dict)

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

    #subjs = {'liqing': 'S1', 'liranran': 'S7', 'longhailiang': 'S6',
    #         'wanghuicui': 'S4', 'zhangdan': 'S3', 'zhangjipeng': 'S2',
    #         'zhuzhiyuan': 'S5'}

    seq = get_emo_sequence(root_dir, 'liqing')
    print seq
    #get_roi_ts(root_dir, seq)    
    #get_conn(root_dir)
    #get_rand_conn(root_dir, 1000)
    #get_mvp_group_roi(root_dir)
    #get_trial_tag(root_dir, 'liqing')

