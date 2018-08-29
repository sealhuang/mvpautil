# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import nibabel as nib
from scipy import io as sio

from nitools import roi as niroi
from nitools import base as nibase
from nitools.roi import extract_mean_ts


def refine_rois(root_dir):
    """Refine ROIs."""
    orig_roi_file = os.path.join(root_dir, 'group-level', 'rois', 'neurosynth',
                                'merged_hfdn_mask_Tmax_s2_lmax_roi_orig.nii.gz')
    roi_info_file = os.path.join(root_dir, 'group-level', 'rois', 'neurosynth',
                                 'new_neurosynth_roi_info.csv')
    roi_info = open(roi_info_file, 'r').readlines()
    roi_info = [line.strip().split(',') for line in roi_info]
    roi_info.pop(0)
    # refine rois
    orig_roi = nib.load(orig_roi_file).get_data()
    new_roi = np.zeros_like(orig_roi)
    for line in roi_info:
        oid = int(line[1])
        nid = int(line[0])
        new_roi[orig_roi==oid] = nid
    # save file
    new_roi_file = os.path.join(root_dir, 'group-level', 'rois', 'neurosynth',
                                'merged_hfdn_mask_Tmax_s2_lmax_roi.nii.gz')
    aff = nib.load(orig_roi_file).affine
    nibase.save2nifti(new_roi, aff, new_roi_file)

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
    os.system(' '.join(str_cmd))
    beta_file = os.path.join(root_dir, 'workshop', 'glmmodel', 'betas', sid,
                           '%s_beta_s2_full.nii.gz'%(sid))
    mni_beta_file = os.path.join(subj_dir, '%s_beta_s2_full_mni.nii.gz'%(sid))
    str_cmd = ['flirt', '-in', beta_file, '-ref', mni_vol, '-applyxfm', '-init',
               func2mni_mat, '-out', mni_beta_file]
    os.system(' '.join(str_cmd))

def get_emo_ts(root_dir, sid, seq):
    """Get neural activity time course of each roi on each emotion condition."""
    subj_dir = os.path.join(root_dir, 'workshop', 'glmmodel', 'conn', sid)

    # load roi and betas
    rois = nib.load(os.path.join(root_dir, 'group-level', 'rois', 'neurosynth',
                    'merged_hfdn_mask_Tmax_s2_lmax_roi.nii.gz')).get_data()
    roi_num = int(rois.max())
    beta1_file = os.path.join(subj_dir, '%s_beta_s1_full_mni.nii.gz'%(sid))
    beta2_file = os.path.join(subj_dir, '%s_beta_s2_full_mni.nii.gz'%(sid))
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
            for r in range(roi_num):
                run_roi_ts[e, r] = niroi.extract_mean_ts(emo_beta, rois==(r+1))
        roi_ts[:, :, (i*20):(i*20+20)] = run_roi_ts

    outfile = os.path.join(subj_dir, 'roi_ts.npy')
    np.save(outfile, roi_ts)

def get_emo_std_ts(root_dir, sid, seq):
    """Get neural activity time course of each roi on each emotion condition."""
    subj_dir = os.path.join(root_dir, 'workshop', 'glmmodel', 'conn', sid)

    # load roi and betas
    rois = nib.load(os.path.join(root_dir, 'group-level', 'rois', 'power264',
                    'power264_rois.nii.gz')).get_data()
    roi_num = int(rois.max())
    beta1_file = os.path.join(subj_dir, '%s_beta_s1_full_mni.nii.gz'%(sid))
    beta2_file = os.path.join(subj_dir, '%s_beta_s2_full_mni.nii.gz'%(sid))
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
        # data normalization for each run
        m = np.mean(run_beta, axis=3, keepdims=True)
        s = np.std(run_beta, axis=3, keepdims=True)
        run_beta = (run_beta - m) / (s + 1e-5)
        # get trial sequence for each emotion
        run_roi_ts = np.zeros((4, roi_num, 20))
        for e in range(4):
            emo_seq = seq[i][e]
            emo_beta = run_beta[..., emo_seq]
            # get time course for each roi
            for r in range(roi_num):
                run_roi_ts[e, r] = niroi.extract_mean_ts(emo_beta, rois==(r+1))
        roi_ts[:, :, (i*20):(i*20+20)] = run_roi_ts
    outfile = os.path.join(subj_dir, 'roi_std_ts_264.npy')
    np.save(outfile, roi_ts)
    outfile = os.path.join(subj_dir, 'roi_std_ts_264.mat')
    sio.savemat(outfile, {'roi_ts': roi_ts})

def get_conn(root_dir, sid):
    """Get connectivity matrix."""
    subj_dir = os.path.join(root_dir, 'workshop', 'glmmodel', 'conn', sid)
    roi_ts_file = os.path.join(subj_dir, 'roi_std_ts.npy')
    roi_ts = np.load(roi_ts_file)
    emo_num = roi_ts.shape[0]
    roi_num = roi_ts.shape[1]

    emo_conn = np.zeros((emo_num, roi_num, roi_num))
    for e in range(emo_num):
        emo_conn[e] = np.corrcoef(roi_ts[e])
    np.save(os.path.join(subj_dir, 'roi_std_conn.npy'), emo_conn)
    #outfile = os.path.join(ppi_dir, 'conn_mtx.mat')
    #sio.savemat(outfile, conn_dict)

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

def power264roi(root_dir):
    """Make ROI file based on Power264 atlas."""
    roi_info = open('power264.csv').readlines()
    roi_info.pop(0)
    roi_info = [line.strip().split(',') for line in roi_info]
    roi_dict = {}
    roi_label_dict = {}
    for line in roi_info:
        if not line[5] in roi_dict:
            roi_dict[line[5]] = {}
            roi_label_dict[line[5]] = {}
        i = int((90.0 - int(line[2])) / 2)
        j = int((int(line[3]) + 126) / 2)
        k = int((int(line[4]) + 72) / 2)
        roi_dict[line[5]][int(line[0])] = [i, j, k]
        roi_label_dict[line[5]][int(line[0])] = line[1]
    
    # create cube roi based on center coord
    #for m in roi_dict:
    #    centers = roi_dict[m]
    #    mask = np.zeros((91, 109, 91))
    #    for c in centers:
    #        mask = niroi.cube_roi(mask, centers[c][0], centers[c][1],
    #                              centers[c][2], 2, c)
    #    mni_vol = os.path.join(os.environ['FSL_DIR'], 'data', 'standard',
    #                           'MNI152_T1_2mm_brain.nii.gz')
    #    aff = nib.load(mni_vol).affine
    #    outfile ='power264_%s_rois.nii.gz'%(m.replace('/','-').replace(' ','-'))
    #    nibase.save2nifti(mask, aff, outfile)
    
    #sel_module = ['Salience', 'Visual', 'Subcortical',
    #              'Cingulo-opercular Task Control', 'Default mode',
    #              'Fronto-parietal Task Control']
    sel_module = roi_dict.keys()
    froi = open('power264_roi.csv', 'w')
    froi.write('RID,FSL_label,X,Y,Z,Module\n')
    count = 1
    mask = np.zeros((91, 109, 91))
    for m in sel_module:
        centers = roi_dict[m]
        labels = roi_label_dict[m]
        for c in centers:
            x = centers[c][0]
            y = centers[c][1]
            z = centers[c][2]
            for n_x in range(x-2, x+3):
                for n_y in range(y-2, y+3):
                    for n_z in range(z-2, z+3):
                        try:
                            if mask[n_x, n_y, n_z]>0:
                                mask[n_x, n_y, n_z] = 1000
                            else:
                                mask[n_x, n_y, n_z] = count
                        except:
                            pass
            #mask = niroi.cube_roi(mask, centers[c][0], centers[c][1],
            #                      centers[c][2], 2, count)
            froi.write(','.join([str(count), labels[c], str(centers[c][0]),
                                 str(centers[c][1]), str(centers[c][2]),
                                 m])+'\n')
            count += 1
    froi.close()
    mask[mask==1000] = 0
    mni_vol = os.path.join(os.environ['FSL_DIR'], 'data', 'standard',
                           'MNI152_T1_2mm_brain.nii.gz')
    aff = nib.load(mni_vol).affine
    outfile ='power264_rois.nii.gz'
    nibase.save2nifti(mask, aff, outfile)

def gen_power_roi(root_dir):
    """Make ROI file based on Power264 atlas."""
    roi_info = open('power_rois.csv').readlines()
    roi_info.pop(0)
    roi_info = [line.strip().split(',') for line in roi_info]

    mask = np.zeros((91, 109, 91))
    for line in roi_info:
        i = int((90.0 - float(line[3])) / 2)
        j = int((float(line[4]) + 126) / 2)
        k = int((float(line[5]) + 72) / 2)
        label = int(line[1])
        for n_x in range(i-2, i+3):
            for n_y in range(j-2, j+3):
                for n_z in range(k-2, k+3):
                    try:
                        if mask[n_x, n_y, n_z]>0:
                            mask[n_x, n_y, n_z] = 1000
                        else:
                            mask[n_x, n_y, n_z] = label
                    except:
                        pass
    mask[mask==1000] = 0
    mni_vol = os.path.join(os.environ['FSL_DIR'], 'data', 'standard',
                           'MNI152_T1_2mm_brain.nii.gz')
    aff = nib.load(mni_vol).affine
    outfile ='power_rois.nii.gz'
    nibase.save2nifti(mask, aff, outfile)


if __name__=='__main__':
    root_dir = r'/nfs/diskstation/projects/emotionPro'

    #power264roi(root_dir)
    gen_power_roi(root_dir)

    #refine_rois(root_dir)
    #func2mni(root_dir, 'S6')
    #seq = get_emo_seq(root_dir, 'S7')
    #get_emo_ts(root_dir, 'S5', seq)
    #get_emo_std_ts(root_dir, 'S7', seq)
    #get_conn(root_dir, 'S7')
    #get_rand_conn(root_dir, 1000)

