# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np

root_dir = r'/nfs/diskstation/projects/emotionPro'
beh_dir = os.path.join(root_dir, 'beh')

subjs = {'liqing': 'S1',
         'zhangjipeng': 'S2',
         'zhangdan': 'S3',
         'wanghuicui': 'S4',
         'zhuzhiyuan': 'S5',
         'longhailiang': 'S6',
         'liranran': 'S7'}

prefix = r'trial_record'

beh_mtx = dict()

for subj in subjs:
    mtx = np.zeros((10, 4, 5))
    for i in range(10):
        print 'Subject %s - Run %s'%(subj, i+1)
        rsp_file = os.path.join(beh_dir, '%s_%s_run%s.csv'%(prefix, subj, i+1))
        if not os.path.exists(rsp_file):
            print '%s not found'%(rsp_file)
            continue
        rsp_info = open(rsp_file).readlines()
        rsp_info.pop(0)
        rsp_info = [line.strip().split(',') for line in rsp_info]
        for line in rsp_info:
            true_label = int(line[1]) - 1
            if not line[2]=='NaN':
                rsp_label = int(line[2]) - 1
            else:
                rsp_label = 4
            mtx[i, true_label, rsp_label] = mtx[i, true_label, rsp_label] + 1
    beh_mtx[subjs[subj]] = mtx

outfile = os.path.join(beh_dir, 'beh_mtx')
np.savez(outfile, **beh_mtx)

