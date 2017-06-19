# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import nibabel as nib
import re

from nipytools import base as mybase

def get_raw_file(db_dir, subj, task):
    """
    Return a dictionary containing cope file path for each raw condition.

    """
    if not task in ['dynamic', 'static']:
        print 'Invalid task category (dynamic or static)'
        return None
    #db_dir = r'/nfs/t2/fmricenter/volume'
    #db_dir = r'/nfs/t1/dpcenter/fmri'
    # face: cope4
    # object: cope7
    # scene: cope12
    # scramble: cope8
    fp = {}
    if task == 'dynamic':
        fp['face'] = os.path.join(db_dir, subj, 'obj.gfeat', 'cope4.feat',
                                  'stats', 'cope1.nii.gz')
        fp['object'] = os.path.join(db_dir, subj, 'obj.gfeat', 'cope7.feat',
                                    'stats', 'cope1.nii.gz')
        fp['scene'] = os.path.join(db_dir, subj, 'obj.gfeat', 'cope12.feat',
                                   'stats', 'cope1.nii.gz')
        fp['scramble'] = os.path.join(db_dir, subj, 'obj.gfeat', 'cope8.feat',
                                      'stats', 'cope1.nii.gz')
    else:
        fp['face'] = os.path.join(db_dir, subj, 'fofo_loc.gfeat',
                                  'cope4.feat', 'stats', 'cope1.nii.gz')
        fp['object'] = os.path.join(db_dir, subj, 'fofo_loc.gfeat',
                                    'cope7.feat', 'stats', 'cope1.nii.gz')
        fp['scene'] = os.path.join(db_dir, subj, 'fofo_loc.gfeat',
                                   'cope12.feat', 'stats', 'cope1.nii.gz')
        fp['scramble'] = os.path.join(db_dir, subj, 'fofo_loc.gfeat',
                                      'cope8.feat', 'stats', 'cope1.nii.gz')
    return fp

def get_single_run_cope(run_dir):
    """
    Return a dictionary containing cope file path for each raw condition
    from a single run.

    """
    # face: cope4
    # object: cope7
    # scene: cope12
    # scramble: cope8
    fp = {}
    #-- object localizer
    fp['face'] = os.path.join(run_dir, 'func.feat', 'reg_standard',
                              'stats', 'cope4.nii.gz')
    fp['object'] = os.path.join(run_dir, 'func.feat', 'reg_standard',
                                'stats', 'cope7.nii.gz')
    fp['scene'] = os.path.join(run_dir, 'func.feat', 'reg_standard',
                               'stats', 'cope12.nii.gz')
    fp['scramble'] = os.path.join(run_dir, 'func.feat', 'reg_standard',
                                  'stats', 'cope8.nii.gz')
    #-- fofo_loc task
    #fp['face'] = os.path.join(db_dir, subj, 'fofo_loc.gfeat', 'cope4.feat',
    #                          'stats', 'cope1.nii.gz')
    #fp['object'] = os.path.join(db_dir, subj, 'fofo_loc.gfeat', 'cope7.feat',
    #                          'stats', 'cope1.nii.gz')
    #fp['scene'] = os.path.join(db_dir, subj, 'fofo_loc.gfeat', 'cope12.feat',
    #                          'stats', 'cope1.nii.gz')
    #fp['scramble'] = os.path.join(db_dir, subj, 'fofo_loc.gfeat', 'cope8.feat',
    #                          'stats', 'cope1.nii.gz')
    return fp

def get_label_file(subject_dir):
    """
    Get a subject-specific label file.

    """
    f_list = os.listdir(subject_dir)
    for f in f_list:
        if re.search('_ff.nii.gz', f):
            return os.path.join(subject_dir, f)

def save2nifti(data, output_file):
    """
    Save a nifti file.

    """
    fsl_dir = os.getenv('FSL_DIR')
    template_file = os.path.join(fsl_dir, 'data', 'standard',
                                 'MNI152_T1_2mm_brain.nii.gz')
    header = nib.load(template_file).get_header()
    mybase.save2nifti(data, header, output_file)


