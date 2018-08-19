# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:


import os
from scipy.io import loadmat
import nibabel as nib

from nitools.base import save2nifti


def mat2nii(mat_structure, data_name, template_file, out_file):
    """save `data_name` from `mat_structure` to nii file.
    `data_name` can be one of `r2_train`, `r2_val`, `beta_train`, `beta_val`
    and `hrfs`.
    """
    data = mat_structure[data_name]
    img = nib.load(template_file)
    aff = img.affine
    save2nifti(data, aff, out_file)

if __name__=='__main__':
    root_dir = r'/nfs/diskstation/projects/emotionPro/workshop/glmmodel'
    nii_dir = os.path.join(root_dir, 'nii')
    beta_dir = os.path.join(root_dir, 'betas')

    sid = 'S1'
    
    # for full estimate
    var = ['hrfs', 'beta', 'r2']
    template_file = os.path.join(nii_dir, sid, 'mcsfunc_1.nii.gz')
    for s in [1, 2]:
        mf = os.path.join(beta_dir, sid, '%s_results_s%s_full.mat'%(sid, s))
        mat = loadmat(mf)
        for v in var:
            f = os.path.join(beta_dir, sid, '%s_%s_s%s_full.nii.gz'%(sid, v, s))
            mat2nii(mat, v, template_file, f)

    ## for cross-validation
    #var = ['hrfs', 'beta_train', 'beta_val', 'r2_train', 'r2_val']
    #template_file = os.path.join(nii_dir, sid, 'mcsfunc_1.nii.gz')
    #for s in [1, 2]:
    #    for r in range(1, 6):
    #        mf = os.path.join(beta_dir, sid, '%s_results_s%s_t%s.mat'%(sid,s,r))
    #        mat = loadmat(mf)
    #        for v in var:
    #            f=os.path.join(beta_dir,sid,'%s_%s_s%s_t%s.nii.gz'%(sid,v,s,r))
    #            mat2nii(mat, v, template_file, f)


