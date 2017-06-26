# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import nibabel as nib
import time
import functools
from multiprocessing import Pool

from nitools import base as nibase
from nitools import roi as niroi
import util

def calculate_mvpa(subj, db_dir, mask_coord, out_dir, neighbor_size):
    """
    Calculate MVPA index for each voxel.

    """
    print subj
    raw_file = util.get_raw_file(db_dir, subj, 'dynamic')
    #raw_file = util.get_raw_file(db_dir, subj, 'static')
    face_cope = nib.load(raw_file['face']).get_data()
    object_cope = nib.load(raw_file['object']).get_data()
    scene_cope = nib.load(raw_file['scene']).get_data()
    scramble_cope = nib.load(raw_file['scramble']).get_data()
    # get mean cope across conditions
    mean_cope = (face_cope + object_cope + scene_cope + scramble_cope) / 4
    # remove mean cope
    face_cope = face_cope - mean_cope
    object_cope = object_cope - mean_cope
    scramble_cope = scramble_cope - mean_cope
    # create output data
    r_data = np.zeros((91, 109, 91))
    for c in mask_coord:
        cube_roi = np.zeros((91, 109, 91))
        cube_roi = niroi.cube_roi(cube_roi, c[0], c[1], c[2], neighbor_size, 1)
        cube_coord = niroi.get_roi_coord(cube_roi)
        face_vtr = niroi.get_voxel_value(cube_coord, face_cope)
        object_vtr = niroi.get_voxel_value(cube_coord, object_cope)
        scramble_vtr = niroi.get_voxel_value(cube_coord, scramble_cope)
        #r_data[tuple(c)] = np.corrcoef(face_vtr, object_vtr)[0, 1]
        r_data[tuple(c)] = np.corrcoef(face_vtr, scramble_vtr)[0, 1]
        #r_data[tuple(c)] = np.corrcoef(scramble_vtr, object_vtr)[0, 1]
    r_data[np.isnan(r_data)] = 0
    out_file = os.path.join(out_dir, subj + '_mvpa.nii.gz')
    util.save2nifti(r_data, out_file)

def calculate_mvpa_sess():
    """
    Calculate MVPA index for each voxel.

    """
    base_dir = r'/nfs/t3/workingshop/huanglijie/uni_mul_analysis'
    doc_dir = os.path.join(base_dir, 'doc')
    data_dir = os.path.join(base_dir, 'multivariate', 'neo_mvpa_n2')
    targ_dir = os.path.join(data_dir, 'face_scramble')

    sessid_file = os.path.join(doc_dir, 'sessid_dc')
    sessid = open(sessid_file).readlines()
    sessid = [line.strip() for line in sessid]

    mask_file = os.path.join(data_dir, 'all_brain_mask.nii.gz')
    mask_data = nib.load(mask_file).get_data()
    mask_coord = niroi.get_roi_coord(mask_data)

    cope_db_dir = r'/nfs/t3/workingshop/huanglijie/fmri/face/volume'
    #cope_db_dir = r'/nfs/h2/face_development/fmri'

    pool = Pool(processes=20)
    pool.map(functools.partial(calculate_mvpa, db_dir=cope_db_dir,
                               mask_coord=mask_coord, out_dir=targ_dir,
                               neighbor_size=2), sessid)
    pool.terminate()

def calculate_mvpa_reliability(subj, db_dir, mask_coord, out_dir,
                               neighbor_size, pair):
    """
    Calculate reliability of MVPA index for each voxel.

    """
    print subj
    subj_dir = os.path.join(db_dir, subj, 'obj')
    rlf_file = os.path.join(subj_dir, 'obj.rlf')
    rlf_info = open(rlf_file).readlines()
    rlf_info = [line.strip() for line in rlf_info]
    if pair == 1:
        rlf = [rlf_info[0], rlf_info[1]]
    elif pair == 2:
        rlf = [rlf_info[0], rlf_info[2]]
    elif pair == 3:
        rlf = [rlf_info[1], rlf_info[2]]
    else:
        print 'Invalid pair parameter ...'
        return None
    raw_file_1 = util.get_single_run_cope(os.path.join(subj_dir, rlf[0]))
    face_cope_1 = nib.load(raw_file_1['face']).get_data()
    object_cope_1 = nib.load(raw_file_1['object']).get_data()
    scene_cope_1 = nib.load(raw_file_1['scene']).get_data()
    scramble_cope_1 = nib.load(raw_file_1['scramble']).get_data()
    mean_cope_1 = (face_cope_1+object_cope_1+scene_cope_1+scramble_cope_1) / 4
    raw_file_2 = util.get_single_run_cope(os.path.join(subj_dir, rlf[1]))
    face_cope_2 = nib.load(raw_file_2['face']).get_data()
    object_cope_2 = nib.load(raw_file_2['object']).get_data()
    scene_cope_2 = nib.load(raw_file_2['scene']).get_data()
    scramble_cope_2 = nib.load(raw_file_2['scramble']).get_data()
    mean_cope_2 = (face_cope_2+object_cope_2+scene_cope_2+scramble_cope_2) / 4
    
    for cond in ['face', 'object', 'scene', 'scramble']:
        cope_1 = nib.load(raw_file_1[cond]).get_data()
        cope_2 = nib.load(raw_file_2[cond]).get_data()
        # remove mean cope
        cope_1 = cope_1 - mean_cope_1
        cope_2 = cope_2 - mean_cope_2
        # create output data
        r_data = np.zeros((91, 109, 91))
        for c in mask_coord:
            cube_roi = np.zeros((91, 109, 91))
            cube_roi = niroi.cube_roi(cube_roi, c[0], c[1], c[2],
                                      neighbor_size, 1)
            cube_coord = niroi.get_roi_coord(cube_roi)
            vtr_1 = niroi.get_voxel_value(cube_coord, cope_1)
            vtr_2 = niroi.get_voxel_value(cube_coord, cope_2)
            r_data[tuple(c)] = np.corrcoef(vtr_1, vtr_2)[0, 1]
        r_data[np.isnan(r_data)] = 0
        out_file = os.path.join(out_dir, cond, subj + '.nii.gz')
        util.save2nifti(r_data, out_file)

def calculate_mvpa_reliability_sess():
    """
    Calculate reliability of MVPA index for each voxel.

    """
    base_dir = r'/nfs/h1/workingshop/huanglijie/uni_mul_analysis'
    ana_dir = os.path.join(base_dir, 'multivariate', 'reliability', 'mvpa_n2')
    data_dir = os.path.join(ana_dir, 'run_2_3')
    doc_dir = os.path.join(base_dir, 'doc')

    sessid_file = os.path.join(doc_dir, 'sessid_06_sel')
    sessid = open(sessid_file).readlines()
    sessid = [line.strip() for line in sessid]

    mask_file = os.path.join(ana_dir, 'all_brain_mask.nii.gz')
    mask_data = nib.load(mask_file).get_data()
    mask_coord = niroi.get_roi_coord(mask_data)

    cope_db_dir = r'/nfs/t2/fmricenter/volume'
    pair = 3
    #cope_db_dir = r'/nfs/h2/face_development/fmri'

    pool = Pool(processes=28)
    pool.map(functools.partial(calculate_mvpa_reliability, db_dir=cope_db_dir,
                               mask_coord=mask_coord, out_dir=data_dir,
                               neighbor_size=2, pair=pair), sessid)
    pool.terminate()

def calculate_group_roi_mvpa_reliability():
    """
    Calculate reliability of multi-voxel representation for each object
    category in group fROI for each subject.

    """
    base_dir = r'/nfs/t3/workingshop/huanglijie/uni_mul_analysis'
    roi_dir = os.path.join(base_dir, 'mask')
    doc_dir = os.path.join(base_dir, 'doc')

    mask_file = os.path.join(roi_dir, 'func_group', 'neo_func_mask_258.nii.gz')
    mask_data = nib.load(mask_file).get_data()

    roi_list = {1:'rOFA', 2:'lOFA',
                3:'rFFA', 4:'lFFA',
                7:'rpcSTS', 8:'lpcSTS'}

    sessid_file = os.path.join(doc_dir, 'sessid_06')
    sessid = open(sessid_file).readlines()
    sessid = [line.strip() for line in sessid]

    cope_db_dir = r'/nfs/t3/workingshop/huanglijie/fmri/face/volume'

    for roi in roi_list:
        output_file = 'neo_group_%s_mvpa_reliability.csv'%(roi_list[roi])
        f = open(output_file, 'wb')
        f.write('SID,face_1_2,face_1_3,face_2_3,obj_1_2,obj_1_3,obj_2_3,scene_1_2,scene_1_3,scene_2_3,scram_1_2,scram_1_3,scram_2_3\n')
        # get ROI index
        roi_mask = mask_data.copy()
        roi_mask[roi_mask!=roi] = 0
        roi_mask[roi_mask==roi] = 1
        mask_coord = niroi.get_roi_coord(roi_mask)
        # get subject-specific cope data
        start_time = time.time()
        for subj in sessid:
            print subj
            temp = [subj]
            subj_dir = os.path.join(cope_db_dir, subj, 'obj')
            rlf_file = os.path.join(subj_dir, 'obj.rlf')
            rlf_info = open(rlf_file).readlines()
            rlf_info = [line.strip() for line in rlf_info]
            # get run-1 data
            raw_file_1 = util.get_single_run_cope(
                            os.path.join(subj_dir, rlf_info[0]))
            face_1 = nib.load(raw_file_1['face']).get_data()
            object_1 = nib.load(raw_file_1['object']).get_data()
            scene_1 = nib.load(raw_file_1['scene']).get_data()
            scramble_1 = nib.load(raw_file_1['scramble']).get_data()
            mean_cope_1 = (face_1 + object_1 + scene_1 + scramble_1) / 4
            face_1 = face_1 - mean_cope_1
            object_1 = object_1 - mean_cope_1
            scene_1 = scene_1 - mean_cope_1
            scramble_1 = scramble_1 - mean_cope_1
            # get run-2 data
            raw_file_2 = util.get_single_run_cope(
                            os.path.join(subj_dir, rlf_info[1]))
            face_2 = nib.load(raw_file_2['face']).get_data()
            object_2 = nib.load(raw_file_2['object']).get_data()
            scene_2 = nib.load(raw_file_2['scene']).get_data()
            scramble_2 = nib.load(raw_file_2['scramble']).get_data()
            mean_cope_2 = (face_2 + object_2 + scene_2 + scramble_2) / 4
            face_2 = face_2 - mean_cope_2
            object_2 = object_2 - mean_cope_2
            scene_2 = scene_2 - mean_cope_2
            scramble_2 = scramble_2 - mean_cope_2
            # get run-3 data
            raw_file_3 = util.get_single_run_cope(
                            os.path.join(subj_dir, rlf_info[2]))
            face_3 = nib.load(raw_file_3['face']).get_data()
            object_3 = nib.load(raw_file_3['object']).get_data()
            scene_3 = nib.load(raw_file_3['scene']).get_data()
            scramble_3 = nib.load(raw_file_3['scramble']).get_data()
            mean_cope_3 = (face_3 + object_3 + scene_3 + scramble_3) / 4
            face_3 = face_3 - mean_cope_3
            object_3 = object_3 - mean_cope_3
            scene_3 = scene_3 - mean_cope_3
            scramble_3 = scramble_3 - mean_cope_3

            # calculate mvpa reliability
            face_vtr_1 = niroi.get_voxel_value(mask_coord, face_1)
            object_vtr_1 = niroi.get_voxel_value(mask_coord, object_1)
            scene_vtr_1 = niroi.get_voxel_value(mask_coord, scene_1)
            scramble_vtr_1 = niroi.get_voxel_value(mask_coord, scramble_1)
            face_vtr_2 = niroi.get_voxel_value(mask_coord, face_2)
            object_vtr_2 = niroi.get_voxel_value(mask_coord, object_2)
            scene_vtr_2 = niroi.get_voxel_value(mask_coord, scene_2)
            scramble_vtr_2 = niroi.get_voxel_value(mask_coord, scramble_2)
            face_vtr_3 = niroi.get_voxel_value(mask_coord, face_3)
            object_vtr_3 = niroi.get_voxel_value(mask_coord, object_3)
            scene_vtr_3 = niroi.get_voxel_value(mask_coord, scene_3)
            scramble_vtr_3 = niroi.get_voxel_value(mask_coord, scramble_3)
            tmp_corr = np.corrcoef(face_vtr_1, face_vtr_2)[0, 1]
            if np.isnan(tmp_corr):
                v = 'nan'
            else:
                v = tmp_corr
            temp.append(str(v))
            tmp_corr = np.corrcoef(face_vtr_1, face_vtr_3)[0, 1]
            if np.isnan(tmp_corr):
                v = 'nan'
            else:
                v = tmp_corr
            temp.append(str(v))
            tmp_corr = np.corrcoef(face_vtr_2, face_vtr_3)[0, 1]
            if np.isnan(tmp_corr):
                v = 'nan'
            else:
                v = tmp_corr
            temp.append(str(v))
            tmp_corr = np.corrcoef(object_vtr_1, object_vtr_2)[0, 1]
            if np.isnan(tmp_corr):
                v = 'nan'
            else:
                v = tmp_corr
            temp.append(str(v))
            tmp_corr = np.corrcoef(object_vtr_1, object_vtr_3)[0, 1]
            if np.isnan(tmp_corr):
                v = 'nan'
            else:
                v = tmp_corr
            temp.append(str(v))
            tmp_corr = np.corrcoef(object_vtr_2, object_vtr_3)[0, 1]
            if np.isnan(tmp_corr):
                v = 'nan'
            else:
                v = tmp_corr
            temp.append(str(v))
            tmp_corr = np.corrcoef(scene_vtr_1, scene_vtr_2)[0, 1]
            if np.isnan(tmp_corr):
                v = 'nan'
            else:
                v = tmp_corr
            temp.append(str(v))
            tmp_corr = np.corrcoef(scene_vtr_1, scene_vtr_3)[0, 1]
            if np.isnan(tmp_corr):
                v = 'nan'
            else:
                v = tmp_corr
            temp.append(str(v))
            tmp_corr = np.corrcoef(scene_vtr_2, scene_vtr_3)[0, 1]
            if np.isnan(tmp_corr):
                v = 'nan'
            else:
                v = tmp_corr
            temp.append(str(v))
            tmp_corr = np.corrcoef(scramble_vtr_1, scramble_vtr_2)[0, 1]
            if np.isnan(tmp_corr):
                v = 'nan'
            else:
                v = tmp_corr
            temp.append(str(v))
            tmp_corr = np.corrcoef(scramble_vtr_1, scramble_vtr_3)[0, 1]
            if np.isnan(tmp_corr):
                v = 'nan'
            else:
                v = tmp_corr
            temp.append(str(v))
            tmp_corr = np.corrcoef(scramble_vtr_2, scramble_vtr_3)[0, 1]
            if np.isnan(tmp_corr):
                v = 'nan'
            else:
                v = tmp_corr
            temp.append(str(v))
            f.write(','.join(temp)+'\n')
        print 'cost %s s'%(time.time() - start_time)

def mvpa_data_merge():
    """
    Merge data.

    """
    base_dir = r'/nfs/t3/workingshop/huanglijie/uni_mul_analysis'
    doc_dir = os.path.join(base_dir, 'doc')
    data_dir = os.path.join(base_dir, 'multivariate', 'neo_mvpa_n2')
    contrast_dir = os.path.join(data_dir, 'face_obj')

    sessid_file = os.path.join(doc_dir, 'sessid_dp_ctrl')
    sessid = open(sessid_file).readlines()
    sessid = [line.strip() for line in sessid]

    merged_file = os.path.join(data_dir, 'merged_face_obj_mvpa_dp_ctrl.nii.gz')
    str_cmd = ['fslmerge', '-a', merged_file]
    for subj in sessid:
        temp = os.path.join(contrast_dir, subj + '_mvpa.nii.gz')
        str_cmd.append(temp)
    os.system(' '.join(str_cmd))

def zstat_data_merge():
    """
    Merge data.

    """
    base_dir = r'/nfs/h1/workingshop/huanglijie/uni_mul_analysis'
    doc_dir = os.path.join(base_dir, 'doc')
    analysis_dir = os.path.join(base_dir, 'anat_roi')
    data_dir = r'/nfs/t2/fmricenter/volume'

    sessid_file = os.path.join(doc_dir, 'sessid_06')
    sessid = open(sessid_file).readlines()
    sessid = [line.strip() for line in sessid]

    merged_file = os.path.join(analysis_dir, 'merged_data.nii.gz')
    str_cmd = ['fslmerge', '-a', merged_file]
    for subj in sessid:
        temp = os.path.join(data_dir, subj, 'obj.gfeat', 'cope1.feat',
                            'stats', 'zstat1.nii.gz')
        str_cmd.append(temp)
    os.system(' '.join(str_cmd))

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

def calculate_ssr_sl_mvpa():
    """
    Calculate MVPA index for a ROI.

    """
    base_dir = r'/nfs/h1/workingshop/huanglijie/uni_mul_analysis'
    roi_dir = os.path.join(base_dir, 'roi')
    doc_dir = os.path.join(base_dir, 'doc')

    roi_list = [1, 2, 3, 4, 7, 8, 9, 10]

    sessid_file = os.path.join(doc_dir, 'sessid_06')
    sessid = open(sessid_file).readlines()
    sessid = [line.strip() for line in sessid]

    cope_db_dir = r'/nfs/t2/fmricenter/volume'
    roi_db_dir = r'/nfs/t2/BAA/SSR'

    output_file = os.path.join(roi_dir, 'roi_mvpa.csv')
    f = open(output_file, 'wb')
    f.write('SID,rOFA,lOFA,rFFA,lFFA,rpcSTS,lpcSTS,rpSTS,lpSTS\n')

    for subj in sessid:
        print subj
        temp = [subj]
        start_time = time.time()

        subj_roi_dir = os.path.join(roi_db_dir, subj, 'obj', 'face-object')
        mask_file = util.get_label_file(subj_roi_dir)
        mask_data = nib.load(mask_file).get_data()

        raw_file = util.get_raw_file(cope_db_dir, subj, 'dynamic')
        face_cope = nib.load(raw_file['face']).get_data()
        object_cope = nib.load(raw_file['object']).get_data()
        scene_cope = nib.load(raw_file['scene']).get_data()
        scramble_cope = nib.load(raw_file['scramble']).get_data()
        # get mean cope across conditions
        mean_cope = (face_cope+object_cope+scene_cope+scramble_cope) / 4
        # remove mean cope
        face_cope = face_cope - mean_cope
        object_cope = object_cope - mean_cope

        # calculate mvpa index for each roi
        for roi in roi_list:
            roi_mask = mask_data.copy()
            roi_mask[roi_mask!=roi] = 0
            roi_mask[roi_mask==roi] = 1
            if not roi_mask.sum():
                temp.append('Null')
                continue
            mask_coord = niroi.get_roi_coord(roi_mask)
            index_list = []
            for c in mask_coord:
                cube_roi = np.zeros((91, 109, 91))
                cube_roi = niroi.cube_roi(cube_roi, c[0], c[1], c[2], 1, 1)
                cube_coord = niroi.get_roi_coord(cube_roi)
                face_vtr = niroi.get_voxel_value(cube_coord, face_cope)
                object_vtr = niroi.get_voxel_value(cube_coord, object_cope)
                mvpa_index = np.corrcoef(face_vtr, object_vtr)[0, 1]
                if np.isnan(mvpa_index):
                    index_list.append(0)
                    print 'NaN detected!'
                else:
                    index_list.append(mvpa_index)
            mean_index = np.mean(index_list)
            temp.append(str(mean_index))
        f.write(','.join(temp)+'\n')
        print 'cost %s s'%(time.time() - start_time)

def calculate_ssr_mean_mvpa():
    """
    Calculate mean MVPA index for a ROI.

    """
    base_dir = r'/nfs/h1/workingshop/huanglijie/uni_mul_analysis'
    roi_dir = os.path.join(base_dir, 'roi')
    doc_dir = os.path.join(base_dir, 'doc')

    roi_list = [1, 2, 3, 4, 7, 8, 9, 10]

    sessid_file = os.path.join(doc_dir, 'sessid_06')
    sessid = open(sessid_file).readlines()
    sessid = [line.strip() for line in sessid]

    mvpa_db_dir = os.path.join(base_dir, 'multivariate', 'mvpa_n2')
    roi_db_dir = r'/nfs/t2/BAA/SSR'

    output_file = r'roi_mvpa.csv'
    f = open(output_file, 'wb')
    f.write('SID,rOFA,lOFA,rFFA,lFFA,rpcSTS,lpcSTS,rpSTS,lpSTS\n')

    for subj in sessid:
        print subj
        temp = [subj]
        start_time = time.time()

        subj_roi_dir = os.path.join(roi_db_dir, subj, 'obj', 'face-object')
        mask_file = util.get_label_file(subj_roi_dir)
        mask_data = nib.load(mask_file).get_data()

        mvpa_file = os.path.join(mvpa_db_dir, subj+'_mvpa.nii.gz')
        mvpa_data = nib.load(mvpa_file).get_data()

        # calculate mean mvpa index for each roi
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

def calculate_group_roi_sl_mvpa():
    """
    Calculate MVPA index for a ROI.

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

    cope_db_dir = r'/nfs/h2/fmricenter/volume'

    output_file = os.path.join(roi_dir, 'neo_group_roi_mvpa.csv')
    f = open(output_file, 'wb')
    f.write('SID,rOFA,lOFA,rFFA,lFFA,rpcSTS,lpcSTS\n')

    for subj in sessid:
        print subj
        temp = [subj]
        start_time = time.time()

        raw_file = util.get_raw_file(cope_db_dir, subj, 'dynamic')
        face_cope = nib.load(raw_file['face']).get_data()
        object_cope = nib.load(raw_file['object']).get_data()
        scene_cope = nib.load(raw_file['scene']).get_data()
        scramble_cope = nib.load(raw_file['scramble']).get_data()
        # get mean cope across conditions
        mean_cope = (face_cope+object_cope+scene_cope+scramble_cope) / 4
        # remove mean cope
        face_cope = face_cope - mean_cope
        object_cope = object_cope - mean_cope

        # calculate mvpa index for each roi
        for roi in roi_list:
            roi_mask = mask_data.copy()
            roi_mask[roi_mask!=roi] = 0
            roi_mask[roi_mask==roi] = 1
            if not roi_mask.sum():
                temp.append('Null')
                continue
            mask_coord = niroi.get_roi_coord(roi_mask)
            index_list = []
            for c in mask_coord:
                cube_roi = np.zeros((91, 109, 91))
                cube_roi = niroi.cube_roi(cube_roi, c[0], c[1], c[2], 2, 1)
                cube_coord = niroi.get_roi_coord(cube_roi)
                face_vtr = niroi.get_voxel_value(cube_coord, face_cope)
                object_vtr = niroi.get_voxel_value(cube_coord, object_cope)
                mvpa_index = np.corrcoef(face_vtr, object_vtr)[0, 1]
                if np.isnan(mvpa_index):
                    index_list.append(0)
                    print 'NaN detected!'
                else:
                    index_list.append(mvpa_index)
            mean_index = np.mean(index_list)
            temp.append(str(mean_index))
        f.write(','.join(temp)+'\n')
        print 'cost %s s'%(time.time() - start_time)

def calculate_group_roi_mvpa():
    """
    Calculate MVPA index for a ROI.

    """
    base_dir = r'/nfs/t3/workingshop/huanglijie/uni_mul_analysis'
    roi_dir = os.path.join(base_dir, 'multivariate', 'neo_analysis', 'mask')
    doc_dir = os.path.join(base_dir, 'doc')

    zstat_file = os.path.join(roi_dir, 'face_obj_zstat_06.nii.gz')
    mask_file = os.path.join(roi_dir, 'ventral_roi_165_06.nii.gz')
    zstat_data = nib.load(zstat_file).get_data()
    mask_data = nib.load(mask_file).get_data()
    # zstat threshold
    thres = 2.3
    zstat_bin = zstat_data >= thres
    mask_data = mask_data * zstat_bin

    roi_list = [1, 2, 3, 4]

    sessid_file = os.path.join(doc_dir, 'sessid_06')
    sessid = open(sessid_file).readlines()
    sessid = [line.strip() for line in sessid]

    cope_db_dir = r'/nfs/t3/workingshop/huanglijie/fmri/face/volume'

    output_file = r'neo_group_roi_mvpa.csv'
    f = open(output_file, 'wb')
    f.write('SID,rOFA,lOFA,rFFA,lFFA\n')

    for subj in sessid:
        print subj
        temp = [subj]
        start_time = time.time()

        raw_file = util.get_raw_file(cope_db_dir, subj, 'dynamic')
        face_cope = nib.load(raw_file['face']).get_data()
        object_cope = nib.load(raw_file['object']).get_data()
        scene_cope = nib.load(raw_file['scene']).get_data()
        scramble_cope = nib.load(raw_file['scramble']).get_data()
        # get mean cope across conditions
        mean_cope = (face_cope+object_cope+scene_cope+scramble_cope) / 4
        # remove mean cope
        face_cope = face_cope - mean_cope
        object_cope = object_cope - mean_cope
        scene_cope = scene_cope - mean_cope
        scramble_cope = scramble_cope - mean_cope

        # calculate mvpa index for each roi
        for roi in roi_list:
            roi_mask = mask_data.copy()
            roi_mask[roi_mask!=roi] = 0
            roi_mask[roi_mask==roi] = 1
            if not roi_mask.sum():
                temp.append('Null')
                continue
            mask_coord = niroi.get_roi_coord(roi_mask)
            face_vtr = niroi.get_voxel_value(mask_coord, face_cope)
            object_vtr = niroi.get_voxel_value(mask_coord, object_cope)
            scene_vtr = niroi.get_voxel_value(mask_coord, scene_cope)
            scramble_vtr = niroi.get_voxel_value(mask_coord, scramble_cope)
            mvpa_index = np.corrcoef(face_vtr, object_vtr)[0, 1]
            #mvpa_index = np.corrcoef(face_vtr, scramble_vtr)[0, 1]
            #mvpa_index = np.corrcoef(object_vtr, scramble_vtr)[0, 1]
            #mvpa_index = np.corrcoef(face_vtr, scene_vtr)[0, 1]
            #mvpa_index = np.corrcoef(scene_vtr, scramble_vtr)[0, 1]
            if np.isnan(mvpa_index):
                v = 'nan'
            else:
                v = mvpa_index
            temp.append(str(v))
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

def calculate_roi_mean_mvpa_devel():
    """
    Calculate MVPA index for a ROI.

    """
    base_dir = r'/nfs/h1/workingshop/huanglijie/uni_mul_analysis'
    roi_dir = os.path.join(base_dir, 'multivariate', 'detection',
                           'child_roi')
    data_dir = os.path.join(base_dir, 'multivariate', 'detection',
                            'child_mvpa_data', 'obj_scr')
    doc_dir = os.path.join(base_dir, 'doc')

    sessid_file = os.path.join(doc_dir, 'sessid_adult')
    sessid = open(sessid_file).readlines()
    sessid = [line.strip() for line in sessid]

    output_file = r'dev_ffa_mvpa.csv'
    f = open(output_file, 'wb')
    f.write('SID,rFFA\n')

    for subj in sessid:
        print subj
        roi_file = os.path.join(roi_dir, subj + '_rFFA.nii.gz')
        data_file = os.path.join(data_dir, subj + '_mvpa.nii.gz')

        mask_data = nib.load(roi_file).get_data()
        mask_data[mask_data>0] = 1
        mvpa_data = nib.load(data_file).get_data()

        # calculate mean RD for rFFA
        o = mvpa_data * mask_data
        temp = o.sum() / mask_data.sum()
        f.write(','.join([subj, str(temp)])+'\n')

def calculate_roi_mvpa_devel():
    """
    Calculate MVPA index for a ROI.

    """
    base_dir = r'/nfs/h1/workingshop/huanglijie/uni_mul_analysis'
    roi_dir = os.path.join(base_dir, 'multivariate', 'detection',
                           'child_roi')
    doc_dir = os.path.join(base_dir, 'doc')

    sessid_file = os.path.join(doc_dir, 'sessid_develop_selected')
    sessid = open(sessid_file).readlines()
    sessid = [line.strip() for line in sessid]

    cope_db_dir = r'/nfs/h2/face_development/fmri'
    roi_list = [3]

    output_file = r'dev_roi_mvpa.csv'
    f = open(output_file, 'wb')
    f.write('SID,rFFA\n')

    for subj in sessid:
        print subj
        temp = [subj]
        start_time = time.time()

        mask_file = os.path.join(roi_dir, subj+'_rFFA.nii.gz')
        mask_data = nib.load(mask_file).get_data()

        raw_file = util.get_raw_file(cope_db_dir, subj, 'static')
        face_cope = nib.load(raw_file['face']).get_data()
        object_cope = nib.load(raw_file['object']).get_data()
        scene_cope = nib.load(raw_file['scene']).get_data()
        scramble_cope = nib.load(raw_file['scramble']).get_data()
        # get mean cope across conditions
        mean_cope = (face_cope+object_cope+scene_cope+scramble_cope) / 4
        # remove mean cope
        face_cope = face_cope - mean_cope
        object_cope = object_cope - mean_cope

        # calculate mvpa index for each roi
        for roi in roi_list:
            roi_mask = mask_data.copy()
            roi_mask[roi_mask!=roi] = 0
            roi_mask[roi_mask==roi] = 1
            if not roi_mask.sum():
                temp.append('Null')
                continue
            mask_coord = niroi.get_roi_coord(roi_mask)
            index_list = []
            for c in mask_coord:
                cube_roi = np.zeros((91, 109, 91))
                cube_roi = niroi.cube_roi(cube_roi, c[0], c[1], c[2], 1, 1)
                cube_coord = niroi.get_roi_coord(cube_roi)
                face_vtr = niroi.get_voxel_value(cube_coord, face_cope)
                object_vtr = niroi.get_voxel_value(cube_coord, object_cope)
                mvpa_index = np.corrcoef(face_vtr, object_vtr)[0, 1]
                if np.isnan(mvpa_index):
                    index_list.append(0)
                    print 'NaN detected!'
                else:
                    index_list.append(mvpa_index)
            mean_index = np.mean(index_list)
            temp.append(str(mean_index))
        f.write(','.join(temp)+'\n')
        print 'cost %s s'%(time.time() - start_time)

if __name__ == '__main__':
    #zstat_data_merge()
    #mvpa_data_merge()
    #z2r()
    #calculate_roi_mvpa()
    calculate_group_roi_mvpa()
    #calculate_mvpa_sess()
    #calculate_mvpa_reliability_sess()
    #calculate_roi_mean_mvpa_devel()
    #calculate_roi_mean_mvpa()
    #calculate_group_roi_mean_mvpa()
    #calculate_roi_mvpa_devel()
    #calculate_group_roi_mvpa()
    #calculate_group_roi_mvpa_reliability()

