# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import nibabel as nib
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt

def datamerge():
    base_dir = r'/nfs/h1/workingshop/huanglijie/uni_mul_analysis'
    db_dir = os.path.join(base_dir, 'multivariate', 'detection', 'mvpcluster')

    n_clusters = 20

    sessid_file = os.path.join(base_dir, 'doc', 'sessid_06')
    sessid = open(sessid_file).readlines()
    sessid = [line.strip() for line in sessid]

    for n in range(10, n_clusters):
        n += 1
        merged_file = os.path.join(db_dir, 'merged_cluster_'+str(n)+'.nii.gz')
        cmdstr = ['fslmerge', '-a', merged_file]
        for subj in sessid:
            temp = os.path.join(db_dir, subj+'_cluster_'+str(n)+'.nii.gz')
            cmdstr.append(temp)
        os.system(' '.join(cmdstr))

def intersubjectconsensus():
    """Compute inter-subjects clustering consensus.

    """
    base_dir = r'/nfs/h1/workingshop/huanglijie/uni_mul_analysis'
    db_dir = os.path.join(base_dir, 'multivariate', 'detection', 'mvpcluster')

    n_clusters = 60

    mask_file = os.path.join(base_dir, 'multivariate', 'detection',
                             'mask.nii.gz')
    mask = nib.load(mask_file).get_data()

    for n in range(1, n_clusters):
        n += 1
        merged_file = os.path.join(db_dir, 'merged_cluster_'+str(n)+'.nii.gz')
        merged_data = nib.load(merged_file).get_data()
        n_subjs = merged_data.shape[3]
        mtx = np.zeros((n_subjs, n_subjs))
        for i in range(n_subjs):
            for j in range(n_subjs):
                data_i = merged_data[..., i]
                data_j = merged_data[..., j]
                vtr_i = data_i[np.nonzero(mask)]
                vtr_j = data_j[np.nonzero(mask)]
                tmp = metrics.adjusted_mutual_info_score(vtr_i, vtr_j)
                mtx[i, j] = tmp
        outfile = os.path.join(db_dir, 'consensus_'+str(n)+'.csv')
        np.savetxt(outfile, mtx, delimiter=',')

def gen_mean_consensus():
    """Get mean consensus index for each cluster number.

    """
    base_dir = r'/nfs/h1/workingshop/huanglijie/uni_mul_analysis'
    db_dir = os.path.join(base_dir, 'multivariate', 'detection', 'mvpcluster')

    n_clusters = 60

    outfile = os.path.join(db_dir, 'consensus_summary.csv')
    f = open(outfile, 'wb')
    f.write('n_clusters,mean\n')
    for n in range(1, n_clusters):
        n += 1
        consensus_file = os.path.join(db_dir, 'consensus_'+str(n)+'.csv')
        mtx = np.loadtxt(open(consensus_file, 'rb'), delimiter=',')
        upper_mtx = np.triu(mtx, k=1)
        m = np.sum(upper_mtx) / 19900
        f.write(','.join([str(n), str(m)])+'\n')

def plot_consensus_map(mtx):
    """Plot consensus matrix.

    """
    sns.set(context='paper', font='monospace')

    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(mtx, vmax=.8, square=True)

def intersubjectconsensus_csv(csv_file):
    """Compute inter-subjects clustering cinsensus from csv file.

    """
    label_mtx = np.loadtxt(open(csv_file, 'rb'), delimiter=',')
    n_subjs = label_mtx.shape[0]
    mtx = np.zeros((n_subjs, n_subjs))
    for i in range(n_subjs):
        for j in range(i+1, n_subjs):
            data_i = label_mtx[i]
            data_j = label_mtx[j]
            tmp = metrics.adjusted_mutual_info_score(data_i, data_j)
            mtx[i, j] = tmp
    upper_mtx = np.triu(mtx, k=1)
    m = np.sum(upper_mtx) / 19900
    return m

def get_shuffle_dist():
    """Get mvp-clsuetring null dist.

    """
    base_dir = r'/nfs/h1/workingshop/huanglijie/uni_mul_analysis'
    db_dir = os.path.join(base_dir, 'multivariate', 'detection',
                          'mvpcluster', 'shuffled_mvp')

    n_clusters = 60
    n_iter = 100
    
    mean_consensus = np.zeros((n_iter, n_clusters-1))
    for i in range(1, n_clusters):
        for j in range(n_iter):
            csv_file = os.path.join(db_dir, 'cluster_%d_%d.csv'%(i+1, j))
            mean_consensus[j, i-1] = intersubjectconsensus_csv(csv_file)
    outfile = os.path.join(db_dir, 'consensus_summary.csv')
    np.savetxt(outfile, mean_consensus, delimiter=',')

def get_spatial_clustering_consensus():
    """Get spatial-clsuetring consensus.

    """
    base_dir = r'/nfs/h1/workingshop/huanglijie/uni_mul_analysis'
    db_dir = os.path.join(base_dir, 'multivariate', 'detection',
                          'mvpcluster', 'coord_based')

    n_clusters = 60
    
    mean_consensus = np.zeros((n_clusters-1, 2))
    for i in range(1, n_clusters):
        csv_file = os.path.join(db_dir, 'cluster_%d.csv'%(i+1))
        mean_consensus[i-1, 0] = i+1
        mean_consensus[i-1, 1] = intersubjectconsensus_csv(csv_file)
        print 'cluster %s - %s'%(mean_consensus[i-1, 0], mean_consensus[i-1, 1])
    outfile = os.path.join(db_dir, 'consensus_summary.csv')
    np.savetxt(outfile, mean_consensus, delimiter=',')

def test_func():
    """Get spatial-clsuetring consensus.

    """
    base_dir = r'/nfs/h1/workingshop/huanglijie/uni_mul_analysis'
    db_dir = os.path.join(base_dir, 'multivariate', 'detection',
                          'mvpcluster', 'coord_based')

    csv_file = os.path.join(db_dir, 'cluster_10.csv')
    print intersubjectconsensus_csv(csv_file)


if __name__ == '__main__':
    #datamerge()
    #intersubjectconsensus()
    #gen_mean_consensus()
    get_spatial_clustering_consensus()
    #test_func()

