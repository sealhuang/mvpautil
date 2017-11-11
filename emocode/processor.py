# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import nibabel as nib

from pynit.base import unpack as pyunpack
from nitools import base as nibase
from nitools import roi as niroi


def z2r():
    """
    Do a Fisher r-to-z transform.

    """
    source_dir = r'/nfs/h1/workingshop/huanglijie/uni_mul_analysis/multivariate'
    input = os.path.join(source_dir, 'beh_corr', 'rmet', 'merged_data.nii.gz')
    output = os.path.join(source_dir, 'beh_corr', 'rmet',
                          'merged_data_r2z.nii.gz')
    data = nib.load(input).get_data()
    header = nib.load(input).get_header()
    one_ele = np.sum(data == 1)
    print one_ele
    data[data==1] = 0.999999
    data[data==-1] = -0.999999
    one_ele = np.sum(data == 1)
    print one_ele
    zdata = np.log((1+data)/(1-data)) / 2
    nibase.save2nifti(zdata, header, output)

def get_mvp_group_roi(scanlist_file, mask_file):
    """Get multivoxel activity pattern for each srimulus from each ROI."""
    # get scan info from scanlist
    [scan_info, subj_list] = pyunpack.readscanlist(scanlist_file)
    # directory config
    nii_dir = scan_info['sessdir']
    
    # load rois
    mask_data = nib.load(mask_file).get_data()
    roi_dict = {'rOFA': 1, 'lOFA': 2, 'rFFA': 3, 'lFFA': 4}
 
    #output_file = os.path.join(roi_dir, 'neo_group_roi_mvpa.csv')
    #f = open(output_file, 'wb')
    #f.write('SID,rOFA,lOFA,rFFA,lFFA,rpcSTS,lpcSTS\n')

    for subj in subj_list:
        # get run infor for emo task
        sid = subj.sess_ID
        subj_dir = os.path.join(nii_dir, sid, 'emo')
        # get par index for each emo run
        if not 'emo' in subj.run_info:
            continue
        [run_idx, par_idx] = subj.getruninfo('emo')
        # var for MVP
        mvp_dict = {}
        for roi in roi_dict:
            mvp_dict[roi] = []
        for i in range(len(run_idx)):
            run_dir = os.path.join(subj_dir, '00'+run_idx[i])
            trn_file = os.path.join(run_dir, 'train_merged_cope.nii.gz')
            test_file = os.path.join(run_dir, 'test_merged_cope.nii.gz')
            trn_cope = nib.load(trn_file).get_data()
            test_cope = nib.load(test_file).get_data()
            run_cope = np.concatenate((trn_cope, test_cope), axis=3)
            # XXX: remove mean cope from each trial
            #mean_cope = np.mean(run_cope, axis=3, keepdims=True)
            #run_cope = run_cope - mean_cope
            # get MVP for each ROI
            for roi in roi_dict:
                roi_mask = mask_data.copy()
                roi_mask[roi_mask!=roi_dict[roi]] = 0
                roi_mask[roi_mask==roi_dict[roi]] = 1
                roi_coord = niroi.get_roi_coord(roi_mask)
                for j in range(run_cope.shape[3]):
                    trl_vtr = niroi.get_voxel_value(roi_coord, run_cope[..., j])

        f.write(','.join(temp)+'\n')
        print 'cost %s s'%(time.time() - start_time)

def calculate_group_roi_mean_mvpa():
    """
    Calculate mean MVPA index for a ROI derived from group analysis.

    """
    base_dir = r'/nfs/t3/workingshop/huanglijie/uni_mul_analysis'
    roi_dir = os.path.join(base_dir, 'mask')
    doc_dir = os.path.join(base_dir, 'doc')

    mask_file = os.path.join(roi_dir, 'func_group', 'func_mask_258.nii.gz')
    mask_data = nib.load(mask_file).get_data()

    roi_list = [1, 2, 3, 4, 7, 8]

    sessid_file = os.path.join(doc_dir, 'sessid_06')
    sessid = open(sessid_file).readlines()
    sessid = [line.strip() for line in sessid]

    mvpa_db_dir = os.path.join(base_dir, 'multivariate',
                               'h2_mvpa_n2', 'face_obj')

    output_file = r'group_roi_mean_mvpa.csv'
    f = open(output_file, 'wb')
    f.write('SID,rOFA,lOFA,rFFA,lFFA,rpcSTS,lpcSTS\n')

    for subj in sessid:
        print subj
        temp = [subj]
        start_time = time.time()
        
        mvpa_file = os.path.join(mvpa_db_dir, subj+'_mvpa.nii.gz')
        mvpa_data = nib.load(mvpa_file).get_data()

        # calculate mvpa index for each roi
        for roi in roi_list:
            roi_mask = mask_data == roi
            if not roi_mask.sum():
                temp.append('Null')
                continue
            else:
                mean_index = np.mean(mvpa_data[roi_mask])
                temp.append(str(mean_index))
        f.write(','.join(temp)+'\n')
        print 'cost %s s'%(time.time() - start_time)



if __name__ == '__main__':

