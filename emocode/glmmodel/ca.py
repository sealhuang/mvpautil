# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import nibabel as nib

from nitools import roi as niroi
from nitools import base as nibase



if __name__=='__main__':
    root_dir = r'/nfs/diskstation/projects/emotionPro'
    
    subj = 'S1'
    val_run_id = 5

    # load roi map
    roi_file = os.path.join(root_dir, 'workshop', 'glmmodel', 'nii',
                            subj, 'rois', '%s_face_rois.nii.gz'%(subj))
    
    # load beta maps
    beta_files = ['%s_beta_%s_%s_t%s.nii.gz'%(subj, g, s, val_run_id)
                  for g in ['train', 'val']
                  for s in ['s1', 's2']]
    beta_arrays = []
    for f in beta_files:
        f = os.path.join(root_dir, 'workshop', 'glmmodel', 'betas',
                         subj, f)
        a = nib.load(f).get_data()
        beta_arrays.append(a)


