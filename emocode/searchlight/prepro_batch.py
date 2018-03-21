# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import subprocess

base_dir = r'/nfs/diskstation/projects/emotionPro'
doc_dir = os.path.join(base_dir, 'doc')
nii_dir = os.path.join(base_dir, 'nii')
pro_dir = os.path.join(base_dir, 'prepro')

tmpl = os.path.join(base_dir, 'script', 'prepro_template.sh')

sessid_file = os.path.join(doc_dir, 'sessid')
sessid = open(sessid_file).readlines()
sessid = [line.strip() for line in sessid]

for sid in sessid:
    subj_dir = os.path.join(nii_dir, sid, 'emo')
    rlf = open(os.path.join(subj_dir, 'emo.rlf')).readlines()
    rlf = [line.strip() for line in rlf]
    for run in rlf:
        print 'SID %s -- Run %s'%(sid, run)
        out_f = os.path.join(base_dir, 'script', sid+'_'+run+'.sh')
        replacements = {'XXXX': sid, 'YYYY': run, 'ZZZZ': sid[:3]+'1'}
        with open(tmpl) as infile, open(out_f, 'w') as outfile:
            for line in infile:
                for src, target in replacements.iteritems():
                    line = line.replace(src, target)
                outfile.write(line)
        os.system('chmod 755 %s'%(out_f))
        subprocess.call(out_f, shell=True)

