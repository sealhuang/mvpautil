# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import nibabel as nib
import matplotlib.pylab as plt

from nitools import roi as niroi
from nitools import base as nibase


def act_fmri_corr(betas, roi_mask, acts):
    """Calculate correlation between cnn activations and fmri responses."""
    x, y, z = np.nonzero(roi_mask)
    print '%s voxels in the mask'%(x.shape[0])
    for i in range(x.shape[0]):
        print 'Voxel %s ...'%(i+1)
        vxl_rsp = betas[x[i], y[i], z[i]]
        for j in range(acts.shape[3]):
            corr_mat = np.zeors((6, 6))
            for a in range(6):
                for b in range(6):
                    pos_rsp = acts[:, a, b, j]
                    corr_mat[a, b] = np.corrcoef(vxl_rsp, pos_rsp)[0, 1]
            plt.imshow(corr_mat, interpolation='nearest')
            plt.colorbar()
            plt.savefig('v%s_c%s.png'%(i+1, j+1))
            plt.close()


if __name__=='__main__':
    root_dir = r'/nfs/diskstation/projects/emotionPro/workshop/glmmodel'
    
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

    act_fmri_corr(betas, roi_mask, acts)

