# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

"""This script is used for fmri data pre-processing which ahead of GLMdenoise
pipeline. The procedure performs corrections for slice timing and motion within
one scan session (consists of 5 runs).

"""

import os
import subprocess

from nitools import unpack as pyunpack


def slicetimer(root_dir, sid):
    """Slice timer for one subject."""
    # dir config
    doc_dir = os.path.join(root_dir, 'doc')
    nii_dir = os.path.join(root_dir, 'nii')
    work_dir = os.path.join(root_dir, 'workshop', 'glmmodel')
    subj_dir = os.path.join(work_dir, 'nii', sid)
    if not os.path.exists(subj_dir):
        os.makedirs(subj_dir, 0755)
    scanlist_file = os.path.join(doc_dir, 'scanlist.csv')
    # read scanlist info
    [scan_info, subj_list] = pyunpack.readscanlist(scanlist_file)
    for subj in subj_list:
        if (subj.sess_ID[:2]==sid) and ('emo' in subj.run_info):
            [run_list, par_list] = subj.getruninfo('emo')
            for i in range(len(run_list)):
                src_file = os.path.join(nii_dir, subj.sess_ID, 'emo',
                                        '00'+run_list[i], 'func.nii.gz')
                targ_file = os.path.join(subj_dir,
                                         'orig_func_'+par_list[i]+'.nii.gz')
                cp_data_cmd = ['fslmaths', src_file, targ_file, '-odt', 'float']
                print ' '.join(cp_data_cmd)
                subprocess.call(' '.join(cp_data_cmd), shell=True)
                stc_file = os.path.join(subj_dir,'sfunc_'+par_list[i])
                slice_time_cmd = ['slicetimer', '-i', targ_file, '-o', stc_file,
                                  '-r', '2', '--odd']
                print ' '.join(slice_time_cmd)
                subprocess.call(' '.join(slice_time_cmd), shell=True)

def intra_session_mc(root_dir, sid, session):
    """motion correction across runs within one session."""
    # dir config
    work_dir = os.path.join(root_dir, 'workshop', 'glmmodel')
    subj_dir = os.path.join(work_dir, 'nii', sid)
    if not os.path.exists(subj_dir):
        os.makedirs(subj_dir, 0755)
    mc_dir = os.path.join(subj_dir, 'mc')
    run_list = {1: [1, 2, 3, 4, 5],
                2: [6, 7, 8, 9, 10]}
    sel_runs = run_list[session]
    for i in range(len(sel_runs)):
        src_file = os.path.join(subj_dir, 'sfunc_%s'%(sel_runs[i]))
        ref_vol = os.path.join(subj_dir, 'ref_vol_session%s'%(session))
        # select reference volume from the first run data
        if not i:
            sel_ref_vol_cmd = ['fslroi', src_file, ref_vol, '177', '1']
            print ' '.join(sel_ref_vol_cmd)
            subprocess.call(' '.join(sel_ref_vol_cmd), shell=True)
        # mc process
        mc_file = os.path.join(subj_dir, 'mcsfunc_%s'%(sel_runs[i]))
        mc_cmd = ['mcflirt', '-in', src_file, '-out', mc_file, '-mats',
                  '-plots', '-reffile', ref_vol, '-rmsrel', '-rmsabs',
                  '-sinc_final']
        print ' '.join(mc_cmd)
        subprocess.call(' '.join(mc_cmd), shell=True)
        # move mc files into run-specific dir
        run_mc_dir = os.path.join(mc_dir, 'run%s'%(sel_runs[i]))
        if not os.path.exists(run_mc_dir):
            os.makedirs(run_mc_dir, 0755)
        mc_res = ['.mat', '.par', '_abs.rms', '_abs_mean.rms', '_rel.rms',
                  '_rel_mean.rms']
        mv_cmd = ['mv', '-f']
        for item in mc_res:
            tmp = os.path.join(subj_dir, 'mcsfunc_%s'%(sel_runs[i])+item)
            mv_cmd.append(tmp)
        mv_cmd.append(run_mc_dir)
        print ' '.join(mv_cmd)
        subprocess.call(' '.join(mv_cmd), shell=True)

def inter_session_mc(root_dir, sid, session):
    """motion correction across runs from different sessions."""
    # dir config
    work_dir = os.path.join(root_dir, 'workshop', 'glmmodel')
    subj_dir = os.path.join(work_dir, 'nii', sid)
    if not os.path.exists(subj_dir):
        os.makedirs(subj_dir, 0755)
    mc_dir = os.path.join(subj_dir, 'mc')
    run_list = {1: [1, 2, 3, 4, 5],
                2: [6, 7, 8, 9, 10]}
    sel_runs = run_list[session]
    for i in range(len(sel_runs)):
        src_file = os.path.join(subj_dir, 'sfunc_%s'%(sel_runs[i]))
        # select reference volume from the first run data
        ref_vol = os.path.join(subj_dir, 'ref_vol_session%s'%(session))
        if not i:
            sel_ref_vol_cmd = ['fslroi', src_file, ref_vol, '177', '1']
            print ' '.join(sel_ref_vol_cmd)
            subprocess.call(' '.join(sel_ref_vol_cmd), shell=True)
            if session==2:
                s1_ref_vol = os.path.join(subj_dir, 'ref_vol_session1')
                s2_ref_vol = os.path.join(subj_dir, 'ref_vol_session2')
                coreg_cmd = ['mcflirt', '-in', s2_ref_vol,
                             '-reffile', s1_ref_vol]
                subprocess.call(' '.join(coreg_cmd), shell=True)
        if session==2:
            ref_vol = os.path.join(subj_dir, 'ref_vol_session2_mcf')
        # mc process
        mc_file = os.path.join(subj_dir, 'mcsfunc_%s'%(sel_runs[i]))
        mc_cmd = ['mcflirt', '-in', src_file, '-out', mc_file, '-mats',
                  '-plots', '-reffile', ref_vol, '-rmsrel', '-rmsabs',
                  '-sinc_final']
        print ' '.join(mc_cmd)
        subprocess.call(' '.join(mc_cmd), shell=True)
        # move mc files into run-specific dir
        run_mc_dir = os.path.join(mc_dir, 'run%s'%(sel_runs[i]))
        if not os.path.exists(run_mc_dir):
            os.makedirs(run_mc_dir, 0755)
        mc_res = ['.mat', '.par', '_abs.rms', '_abs_mean.rms', '_rel.rms',
                  '_rel_mean.rms']
        mv_cmd = ['mv', '-f']
        for item in mc_res:
            tmp = os.path.join(subj_dir, 'mcsfunc_%s'%(sel_runs[i])+item)
            mv_cmd.append(tmp)
        mv_cmd.append(run_mc_dir)
        print ' '.join(mv_cmd)
        subprocess.call(' '.join(mv_cmd), shell=True)

def func2anat(root_dir, sid):
    """Generate a linear registration matrix from functional image to
    anatomical image.
    """
    # dir config
    anat_dir = os.path.join(root_dir, 'nii', sid+'P1', '3danat', 'reg_fsl')
    work_dir = os.path.join(root_dir, 'workshop', 'glmmodel')
    subj_dir = os.path.join(work_dir, 'nii', sid)

    func_vol = os.path.join(subj_dir, 'ref_vol_session1.nii.gz')
    t1_vol = os.path.join(anat_dir, 'T1.nii.gz')
    t1brain_vol = os.path.join(anat_dir, 'T1_brain.nii.gz')
    out_mat = os.path.join(subj_dir, 'ref_vol2highres')
    cmd_str = ['epi_reg', '--epi=%s'%(func_vol), '--t1=%s'%(t1_vol),
               '--t1brain=%s'%(t1brain_vol), '--out=%s'%(out_mat)]
    subprocess.call(' '.join(cmd_str), shell=True)

def standard2func(root_dir, sid):
    """Generate a linear registration matrix from standard space (2mm) to
    functional image (native space).
    """
    # dir config
    anat_dir = os.path.join(root_dir, 'nii', sid+'P1', '3danat', 'reg_fsl')
    func_dir = os.path.join(root_dir, 'workshop', 'glmmodel', 'nii', sid)

    # get standard to highres image matrix
    anat2standard_mat = os.path.join(anat_dir, 'highres2standard_2mm.mat')
    standard2anat_mat = os.path.join(anat_dir, 'standard_2mm2highres.mat')
    cmd_str = ['convert_xfm', '-omat', standard2anat_mat,
               '-inverse', anat2standard_mat]
    subprocess.call(' '.join(cmd_str), shell=True)

    # get highres image to func matrix
    func2anat_mat = os.path.join(func_dir, 'ref_vol2highres.mat')
    anat2func_mat = os.path.join(func_dir, 'highres2ref_vol.mat')
    cmd_str = ['convert_xfm', '-omat', anat2func_mat, '-inverse', func2anat_mat]
    subprocess.call(' '.join(cmd_str), shell=True)

    # get standard to func image
    standard2func_mat = os.path.join(func_dir, 'standard2ref_vol.mat')
    cmd_str = ['convert_xfm', '-omat', standard2func_mat, '-concat',
               anat2func_mat, standard2anat_mat]
    subprocess.call(' '.join(cmd_str), shell=True)


if __name__=='__main__':
    root_dir = r'/nfs/diskstation/projects/emotionPro'
    #slicetimer(root_dir, 'S1')
    #intra_session_mc(root_dir, 'S1', 1)
    #inter_session_mc(root_dir, 'S1', 1)
    #func2anat(root_dir, 'S1')
    standard2func(root_dir, 'S1')

