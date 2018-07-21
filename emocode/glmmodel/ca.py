# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
from scipy.io import savemat
import nibabel as nib
import matplotlib.pylab as plt

from nitools import roi as niroi
from nitools import base as nibase


def save2mat(betas, roi_mask, acts):
    """Save cnn activations and fmri responses to numpy array"""
    # get fmri data
    x, y, z = np.nonzero(roi_mask)
    print '%s voxels in the mask'%(x.shape[0])
    voxel_rsps = np.zeros((betas.shape[3], x.shape[0]))
    for i in range(x.shape[0]):
        vxl_rsps[:, i] = betas[x[i], y[i], z[i]]
    # get cnn data
    cnn_rsps = np.zeros((acts.shape[0], acts.shape[3]))
    for i in range(acts.shape[3]):
        tmp_act = acts[..., j]
        tmp_act = tmp_act.reshape((800, 36)).sum(axis=1)
        cnn_rsps[:, i] = tmp_act
    savemat('ffa_cnn_rsp.mat', {'vxl_rsp': vxl_rsp, 'cnn_rsp': cnn_rsp})

def act_fmri_corr(betas, roi_mask, acts):
    """Calculate correlation between cnn activations and fmri responses."""
    x, y, z = np.nonzero(roi_mask)
    print '%s voxels in the mask'%(x.shape[0])
    for i in range(x.shape[0]):
        print 'Voxel %s ...'%(i+1)
        vxl_rsp = betas[x[i], y[i], z[i]]
        #corr_mat = np.zeros((6, 6, 256))
        corr_mat = np.zeros((256,))
        for j in range(acts.shape[3]):
            tmp_act = acts[..., j]
            tmp_act = tmp_act.reshape((800, 36)).sum(axis=1)
            r = np.corrcoef(vxl_rsp, tmp_act)[0, 1]
            if not np.isnan(r):
                corr_mat[j] = r
            #for a in range(6):
            #    for b in range(6):
            #        pos_rsp = acts[:, a, b, j]
            #        r = np.corrcoef(vxl_rsp, pos_rsp)[0, 1]
            #        if not np.isnan(r):
            #            corr_mat[a, b, j] = r
        #tmp = np.zeros((6, 6))
        #for a in range(6):
        #    for b in range(6):
        #        r = corr_mat[a, b]
        #        tmp[a, b] = np.median(r)
        #plt.imshow(tmp, interpolation=None)
        #corr_mat = corr_mat.max(axis=2)
        #plt.imshow(corr_mat, interpolation=None)
        #plt.colorbar()
        plt.bar(range(1, 257), corr_mat)
        plt.savefig('voxel%s.png'%(i+1))
        plt.close()


if __name__=='__main__':
    root_dir = r'/nfs/diskstation/projects/emotionPro/workshop/glmmodel'

    # analysis info
    sid = 'S1'
    val_run_id = 5

    # load roi map
    roi_file = os.path.join(root_dir, 'nii', sid, 'rois',
                            '%s_face_rois.nii.gz'%(sid))
    roi_data = nib.load(roi_file).get_data()
    roi_mask = roi_data==3
    
    # load beta maps
    beta_files = ['%s_beta_%s_%s_t%s.nii.gz'%(sid, g, s, val_run_id)
                  for s in ['s1', 's2'] for g in ['train', 'val']]
    beta_datas = []
    for f in beta_files:
        f = os.path.join(root_dir, 'betas', sid, f)
        a = nib.load(f).get_data()
        beta_datas.append(a)
    betas = np.concatenate(tuple(beta_datas), axis=3)

    # load cnn activation
    act_files = ['%s_stimuli_%s_pool5.npy'%(sid, i+1) for i in range(10)]
    act_datas = []
    for f in act_files:
        f = os.path.join(root_dir, 'cnncorr', 'emoNet', 'pool5', f)
        a = np.load(f)
        act_datas.append(a)
    acts = np.concatenate(tuple(act_datas), axis=0)

    # save raw data
    save2mat(betas, roi_mask, acts)

    # calculate correlation between cnn activation and fmri
    #act_fmri_corr(betas, roi_mask, acts)

