# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import nibabel as nib
from sklearn.cluster import KMeans

from nipytools import base as mybase

def get_rd_data(db_dir, subj):
    """Get representational dissimilarity (RD) data from DB dir for
    each subject.

    """
    rd_maps = {}
    rd_maps['face_obj'] = os.path.join(db_dir, 'face_obj', subj+'_mvpa.nii.gz')
    rd_maps['face_scr'] = os.path.join(db_dir, 'face_scr', subj+'_mvpa.nii.gz')
    rd_maps['face_scn'] = os.path.join(db_dir, 'face_scn', subj+'_mvpa.nii.gz')
    rd_maps['obj_scr'] = os.path.join(db_dir, 'obj_scr', subj+'_mvpa.nii.gz')
    rd_maps['obj_scn'] = os.path.join(db_dir, 'obj_scn', subj+'_mvpa.nii.gz')
    rd_maps['scn_scr'] = os.path.join(db_dir, 'scn_scr', subj+'_mvpa.nii.gz')
    return rd_maps

def kmeanclustering(data_mtx, n_clusters):
    """Clustering voxels based on MVPs.

    """
    k_means = KMeans(n_clusters)
    k_means.fit(data_mtx)
    
    # center voxel
    #print k_means.cluster_centers_
    
    # clustering inertia
    print k_means.inertia_

    return k_means.labels_

def write2nifti(file_name, voxel_coord, labels):
    """Write label info into nifti files.

    """
    labels = labels + 1
    
    data = np.zeros((91, 109, 91), dtype=np.uint8)
    for i in range(len(voxel_coord)):
        data[voxel_coord[i][0],
             voxel_coord[i][1],
             voxel_coord[i][2]] = labels[i]

    fsl_dir = os.getenv('FSL_DIR')
    std_brain = os.path.join(fsl_dir, 'data', 'standard',
                             'MNI152_T1_2mm_brain.nii.gz')
    header = nib.load(std_brain).get_header()
    header['datatype'] = 4
    mybase.save2nifti(data, header, file_name)

def mvpclustering():
    """Main function.

    """
    base_dir = r'/nfs/h1/workingshop/huanglijie/uni_mul_analysis'
    db_dir = os.path.join(base_dir, 'multivariate', 'detection', 'mvpa_data')

    n_clusters = 60
    out_dir = os.path.join(base_dir, 'multivariate', 'detection', 'mvpcluster')

    mask_file = os.path.join(base_dir, 'multivariate', 'detection',
                             'mask.nii.gz')
    mask_data = nib.load(mask_file).get_data()
    x, y, z = np.nonzero(mask_data)
    voxel_coord = [[x[i], y[i], z[i]] for i in range(len(x))]

    coord_feature_x = (x - x.min()) * 1.0 / (x.max() - x.min())
    coord_feature_y = (y - y.min()) * 1.0 / (y.max() - y.min())
    coord_feature_z = (z - z.min()) * 1.0 / (z.max() - z.min())
    coord_feature = [[2*coord_feature_x[i]-1,
                      2*coord_feature_y[i]-1,
                      2*coord_feature_z[i]-1] for i in range(len(x))]

    sessid_file = os.path.join(base_dir, 'doc', 'sessid_06')
    sessid = open(sessid_file).readlines()
    sessid = [line.strip() for line in sessid]

    for subj in sessid:
        print subj
        rd_maps = get_rd_data(db_dir, subj)
        rd_face_obj = nib.load(rd_maps['face_obj']).get_data()
        rd_face_scr = nib.load(rd_maps['face_scr']).get_data()
        rd_face_scn = nib.load(rd_maps['face_scn']).get_data()
        rd_obj_scr = nib.load(rd_maps['obj_scr']).get_data()
        rd_obj_scn = nib.load(rd_maps['obj_scn']).get_data()
        rd_scn_scr = nib.load(rd_maps['scn_scr']).get_data()
        
        data_mtx = []
        for i in range(len(voxel_coord)):
            temp = []
            temp.append(rd_face_obj[voxel_coord[i][0],
                                    voxel_coord[i][1],
                                    voxel_coord[i][2]])
            temp.append(rd_face_scr[voxel_coord[i][0],
                                    voxel_coord[i][1],
                                    voxel_coord[i][2]])
            temp.append(rd_face_scn[voxel_coord[i][0],
                                    voxel_coord[i][1],
                                    voxel_coord[i][2]])
            temp.append(rd_obj_scr[voxel_coord[i][0],
                                   voxel_coord[i][1],
                                   voxel_coord[i][2]])
            temp.append(rd_obj_scn[voxel_coord[i][0],
                                   voxel_coord[i][1],
                                   voxel_coord[i][2]])
            temp.append(rd_scn_scr[voxel_coord[i][0],
                                   voxel_coord[i][1],
                                   voxel_coord[i][2]])
            temp.append(coord_feature[i][0])
            temp.append(coord_feature[i][1])
            temp.append(coord_feature[i][2])
            data_mtx.append(temp)

        data_mtx = np.array(data_mtx)

        # Cluster voxels into n groups
        for n in range(40, n_clusters):
            n += 1
            print 'Cluster #: %d'%(n)
            labels = kmeanclustering(data_mtx, n)
            outfile = os.path.join(out_dir, subj+'_cluster_'+str(n)+'.nii.gz')
            write2nifti(outfile, voxel_coord, labels)

def shuffle_clustering():
    """Main function for getting shuffled clustering..

    """
    base_dir = r'/nfs/h1/workingshop/huanglijie/uni_mul_analysis'
    db_dir = os.path.join(base_dir, 'multivariate', 'detection', 'mvpa_data')

    n_clusters = 60
    n_iter = 100

    out_dir = os.path.join(base_dir, 'multivariate', 'detection',
                           'mvpcluster', 'shuffled_mvp')
    if not os.path.exists(out_dir):
        os.system('mkdir '+out_dir)

    mask_file = os.path.join(base_dir, 'multivariate', 'detection',
                             'mask.nii.gz')
    mask_data = nib.load(mask_file).get_data()
    x, y, z = np.nonzero(mask_data)
    voxel_coord = [[x[i], y[i], z[i]] for i in range(len(x))]

    coord_feature_x = (x - x.min()) * 1.0 / (x.max() - x.min())
    coord_feature_y = (y - y.min()) * 1.0 / (y.max() - y.min())
    coord_feature_z = (z - z.min()) * 1.0 / (z.max() - z.min())
    coord_feature = [[2*coord_feature_x[i]-1,
                      2*coord_feature_y[i]-1,
                      2*coord_feature_z[i]-1] for i in range(len(x))]

    sessid_file = os.path.join(base_dir, 'doc', 'sessid_06')
    sessid = open(sessid_file).readlines()
    sessid = [line.strip() for line in sessid]

    for sidx in range(n_iter):
        print 'Shuffle # %d'%(sidx)
        for subj in sessid:
            print subj
            rd_maps = get_rd_data(db_dir, subj)
            rd_face_obj = nib.load(rd_maps['face_obj']).get_data()
            rd_face_scr = nib.load(rd_maps['face_scr']).get_data()
            rd_face_scn = nib.load(rd_maps['face_scn']).get_data()
            rd_obj_scr = nib.load(rd_maps['obj_scr']).get_data()
            rd_obj_scn = nib.load(rd_maps['obj_scn']).get_data()
            rd_scn_scr = nib.load(rd_maps['scn_scr']).get_data()

            data_mtx = []
            for i in range(len(voxel_coord)):
                temp = []
                temp.append(rd_face_obj[voxel_coord[i][0],
                                        voxel_coord[i][1],
                                        voxel_coord[i][2]])
                temp.append(rd_face_scr[voxel_coord[i][0],
                                        voxel_coord[i][1],
                                        voxel_coord[i][2]])
                temp.append(rd_face_scn[voxel_coord[i][0],
                                        voxel_coord[i][1],
                                        voxel_coord[i][2]])
                temp.append(rd_obj_scr[voxel_coord[i][0],
                                       voxel_coord[i][1],
                                       voxel_coord[i][2]])
                temp.append(rd_obj_scn[voxel_coord[i][0],
                                       voxel_coord[i][1],
                                       voxel_coord[i][2]])
                temp.append(rd_scn_scr[voxel_coord[i][0],
                                       voxel_coord[i][1],
                                       voxel_coord[i][2]])
                temp.append(coord_feature[i][0])
                temp.append(coord_feature[i][1])
                temp.append(coord_feature[i][2])
                data_mtx.append(temp)

            data_mtx = np.array(data_mtx)
            np.random.shuffle(data_mtx[:, 0])
            np.random.shuffle(data_mtx[:, 1])
            np.random.shuffle(data_mtx[:, 2])
            np.random.shuffle(data_mtx[:, 3])
            np.random.shuffle(data_mtx[:, 4])
            np.random.shuffle(data_mtx[:, 5])

            # Cluster voxels into n groups
            for n in range(1, n_clusters):
                n += 1
                print 'Cluster #: %d'%(n)
                labels = kmeanclustering(data_mtx, n)
                outfile = os.path.join(out_dir, 'cluster_%s_%s.csv'%(n, sidx))
                with open(outfile, 'a') as myfile:
                    myfile.write(','.join([str(item) for item in labels])+'\n')

def spatial_clustering():
    """Main function for getting coordinate-based clustering..

    """
    base_dir = r'/nfs/h1/workingshop/huanglijie/uni_mul_analysis'
    db_dir = os.path.join(base_dir, 'multivariate', 'detection', 'mvpa_data')

    n_clusters = 60

    out_dir = os.path.join(base_dir, 'multivariate', 'detection',
                           'mvpcluster', 'coord_based')
    if not os.path.exists(out_dir):
        os.system('mkdir '+out_dir)

    mask_file = os.path.join(base_dir, 'multivariate', 'detection',
                             'mask.nii.gz')
    mask_data = nib.load(mask_file).get_data()
    x, y, z = np.nonzero(mask_data)
    voxel_coord = [[x[i], y[i], z[i]] for i in range(len(x))]

    coord_feature_x = (x - x.min()) * 1.0 / (x.max() - x.min())
    coord_feature_y = (y - y.min()) * 1.0 / (y.max() - y.min())
    coord_feature_z = (z - z.min()) * 1.0 / (z.max() - z.min())
    coord_feature = [[2*coord_feature_x[i]-1,
                      2*coord_feature_y[i]-1,
                      2*coord_feature_z[i]-1] for i in range(len(x))]

    sessid_file = os.path.join(base_dir, 'doc', 'sessid_06')
    sessid = open(sessid_file).readlines()
    sessid = [line.strip() for line in sessid]

    for subj in sessid:
        print subj
        data_mtx = np.array(coord_feature)

        # Cluster voxels into n groups
        for n in range(1, n_clusters):
            n += 1
            print 'Cluster #: %d'%(n)
            labels = kmeanclustering(data_mtx, n)
            outfile = os.path.join(out_dir, 'cluster_%s.csv'%(n))
            with open(outfile, 'a') as myfile:
                myfile.write(','.join([str(item) for item in labels])+'\n')


if __name__ == '__main__':
    mvpclustering()
    #shuffle_clustering()

