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
    work_dir = os.path.join(root_dir, 'workshop', 'brs')
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
                                         'orig_func_'+par_list[i]+'.nii.gz'))
                cp_data_cmd = ['fslmaths', src_file, targ_file, '-odt', 'float']
                print ' '.join(cp_data_cmd)
                subprocess.call(' '.join(cp_data_cmd), shell=True)
                stc_file = os.path.join(subj_dir,'sfunc_'+par_list[i]+'.nii.gz')
                slice_time_cmd = ['slicetimer', '-i', targ_file, '-o', stc_file,
                                  '-r', '2', '--odd']
                print ' '.join(slice_time_cmd)
                subprocess.call(' '.join(slice_time_cmd), shell=True)


if __name__=='__main__':
    root_dir = r'/nfs/diskstation/projects/emotionPro'
    slicetimer(root_dir, sid)

#doc_dir = os.path.join(base_dir, 'doc')
#nii_dir = os.path.join(base_dir, 'nii')
#pro_dir = os.path.join(base_dir, 'prepro')
#
#tmpl = os.path.join(base_dir, 'script', 'prepro_template.sh')
#
#sessid_file = os.path.join(doc_dir, 'sessid')
#sessid = open(sessid_file).readlines()
#sessid = [line.strip() for line in sessid]
#
#for sid in sessid:
#    subj_dir = os.path.join(nii_dir, sid, 'emo')
#    rlf = open(os.path.join(subj_dir, 'emo.rlf')).readlines()
#    rlf = [line.strip() for line in rlf]
#    for run in rlf:
#        print 'SID %s -- Run %s'%(sid, run)
#        out_f = os.path.join(base_dir, 'script', sid+'_'+run+'.sh')
#        replacements = {'XXXX': sid, 'YYYY': run, 'ZZZZ': sid[:3]+'1'}
#        with open(tmpl) as infile, open(out_f, 'w') as outfile:
#            for line in infile:
#                for src, target in replacements.iteritems():
#                    line = line.replace(src, target)
#                outfile.write(line)
#        os.system('chmod 755 %s'%(out_f))
#        subprocess.call(out_f, shell=True)

