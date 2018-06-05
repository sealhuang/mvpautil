# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:


import os
from scipy.io import loadmat
import nibabel as nib

from nitools.base import save2nifti


def mat2nii(mat_structure, data_name, template_file, out_file):
    """save `data_name` from `mat_structure` to nii file.
    `data_name` can be one of `R2`, `betas`, and `hrfs`.
    """
    data = mat_structure[data_name]
    img = nib.load(template_file)
    aff = img.affine
    save2nifti(data, aff, out_file)

if __name__=='__main__':
    root_dir = r'/nfs/diskstation/projects/emotionPro/workshop/glmdenoise'
    nii_dir = os.path.join(root_dir, 'nii')
    template_file = os.path.join(nii_dir, 'S1', 'mcsfunc_1.nii.gz')
    mat_file = os.path.join(root_dir, 'script', 'train_results_s1.mat')
    mat = loadmat(mat_file)
    mat2nii(mat, 'R2', template_file, 'train_R2_s1.nii.gz') 


