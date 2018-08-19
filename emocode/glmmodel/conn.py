# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import nibabel as nib
from scipy import io as sio

from pynit.base import unpack as pyunpack
from nitools import roi as niroi
from nitools.roi import extract_mean_ts


def get_emo_seq(root_dir, sid):
    """Get trial indexes for each emotion condition for each run."""
    beh_dir = os.path.join(root_dir, 'beh')
    # get subject name
    subj_name = {'S1': 'liqing', 'S2': 'zhangjipeng', 'S3': 'zhangdan',
                 'S4': 'wanghuicui', 'S5': 'zhuzhiyuan', 'S6': 'longhailiang',
                 'S7': 'liranran'}
    subj = subj_name[sid]
    # seq var
    seq = []
    # get trial indexes for each run 
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
        # get trial indexes for each emotion type
        trial_idx = []
        for e in range(4):
            trial_idx.append(np.nonzero(np.array(stim_label)==(e+1))[0])
        seq.append(trial_idx)
    return seq

def func2mni(root_dir, sid):
    """Convert functional data from original space into standard space."""
    conn_dir = os.path.join(root_dir, 'workshop', 'glmmodel', 'conn')
    subj_dir = os.path.join(conn_dir, sid)
    if not os.path.exists(subj_dir):
        os.system('mkdir %s'%(subj_dir))
    # generate registration mat
    func2anat_mat = os.path.join(root_dir, 'workshop', 'glmmodel', 'nii',
                                 sid, 'ref_vol2highres.mat')
    anat2mni_mat = os.path.join(root_dir, 'nii', sid+'P1', '3danat', 'reg_fsl',
                                'highres2standard_2mm.mat')
    func2mni_mat = os.path.join(subj_dir, 'func2standard_2mm.mat')
    str_cmd = ['convert_xfm', '-omat', func2mni_mat, '-concat',
               anat2mni_mat, func2anat_mat]
    os.system(' '.join(str_cmd))
    # convert func images
    mni_vol = os.path.join(os.environ['FSL_DIR'], 'data', 'standard',
                           'MNI152_T1_2mm_brain.nii.gz')
    beta_file = os.path.join(root_dir, 'workshop', 'glmmodel', 'betas', sid,
                           '%s_beta_s1_full.nii.gz'%(sid))
    mni_beta_file = os.path.join(subj_dir, '%s_beta_s1_full_mni.nii.gz'%(sid))
    str_cmd = ['flirt', '-in', beta_file, '-ref', mni_vol, '-applyxfm', '-init',
               func2mni_mat, '-out', mni_beta_file]
    beta_file = os.path.join(root_dir, 'workshop', 'glmmodel', 'betas', sid,
                           '%s_beta_s2_full.nii.gz'%(sid))
    mni_beta_file = os.path.join(subj_dir, '%s_beta_s2_full_mni.nii.gz'%(sid))
    str_cmd = ['flirt', '-in', beta_file, '-ref', mni_vol, '-applyxfm', '-init',
               func2mni_mat, '-out', mni_beta_file]

def get_emo_ts(root_dir, sid, seq):
    """Get neural activity time course of each roi on each emotion condition."""
    subj_dir = os.path.join(root_dir, 'workshop', 'glmmodel', 'conn', sid)

    # load roi and betas
    rois = nib.load(os.path.join(root_dir, 'group-level', 'rois', 'neurosynth',
                    'merged_hfdn_FDR_0.01_Tmax_s2_cuberoi.nii.gz')).get_data()
    roi_num = int(rois.max())
    beta1_file = os.path.join(subj_dir, '%s_beta_s1_full_mni.nii.gz')
    beta2_file = os.path.join(subj_dir, '%s_beta_s2_full_mni.nii.gz')
    beta1 = nib.load(beta1_file).get_data()
    beta2 = nib.load(beta2_file).get_data()

    # get roi time course for each emotion type
    roi_ts = np.zeros((4, roi_num, 200))
    for i in range(10):
        if i<5:
            run_beta = beta1[..., (i*80):(i*80+80)]
        else:
            run_beta = beta2[..., (i*80-400):(i*80-320)]
        print run_beta.shape
        run_roi_ts = np.zeros((4, roi_num, 20))
        # get trial sequence for each emotion
        for e in range(4):
            emo_seq = seq[i][e]
            emo_beta = run_beta[..., emo_seq]
            # get time course for each roi
            for r in roi_num:
                run_roi_ts[e, r] = niroi.extract_mean_ts(emo_beta, rois==(r+1))
        roi_ts[:, :, (i*20):(i*20+20)] = run_roi_ts

    outfile = os.path.join(subj_dir, 'roi_ts.npy')
    np.save(outfile, roi_ts)

def get_conn(root_dir):
    """Get connectivity matrix."""
    ppi_dir = os.path.join(root_dir, 'ppi', 'decovPPI')
    conn_dict = {}
    for i in range(7):
        roi_idx = range(37)
        print 'ROI number: %s'%(len(roi_idx))
        conn_dict['s%s'%(i+1)] = np.zeros((len(roi_idx), len(roi_idx), 4))
        for j in range(4):
            ts = None
            for k in range(10):
                ts_name = r'S%s_roi_ts_run%s_emo%s.npy'%(i+1, k+1, j+1)
                ts_file = os.path.join(ppi_dir, 'roi_ts','rois_meta_r2',ts_name)
                if not os.path.exists(ts_file):
                    print '%s not exists'%(ts_name)
                    continue
                tmp = np.load(ts_file)
                m = tmp.mean(axis=0, keepdims=True)
                s = tmp.std(axis=0, keepdims=True)
                tmp = (tmp - m) / (s + 1e-5)
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
                m = tmp.mean(axis=0, keepdims=True)
                s = tmp.std(axis=0, keepdims=True)
                tmp = (tmp - m) / (s + 1e-5)
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

    roi2func(root_dir, 'S1')
    #seq = get_emo_seq(root_dir, 'S1')
    #print seq
    #get_emo_ts(root_dir, 'S1', seq)
    #get_conn(root_dir)
    #get_rand_conn(root_dir, 1000)
    #get_mvp_group_roi(root_dir)
    #get_trial_tag(root_dir, 'liqing')

