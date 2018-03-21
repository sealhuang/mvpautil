#!/bin/sh

mkdir -p /nfs/diskstation/projects/emotionPro/prepro/XXXX/YYYY
cd /nfs/diskstation/projects/emotionPro/prepro/XXXX/YYYY

fslmaths /nfs/diskstation/projects/emotionPro/nii/XXXX/emo/YYYY/func func_data -odt float
slicetimer -i func_data -o sfunc_data -r 2 --odd
fslroi sfunc_data example_func 177 1
epi_reg --epi=example_func --t1=/nfs/diskstation/projects/emotionPro/nii/ZZZZ/3danat/reg_fsl/T1 --t1brain=/nfs/diskstation/projects/emotionPro/nii/ZZZZ/3danat/reg_fsl/T1_brain --out=example_func2highres
convert_xfm -inverse -omat highres2example_func.mat example_func2highres.mat
mcflirt -in sfunc_data -out sfunc_data_mcf -mats -plots -reffile example_func -rmsrel -rmsabs -spline_final
mkdir -p mc ; /bin/mv -f sfunc_data_mcf.mat sfunc_data_mcf.par sfunc_data_mcf_abs.rms sfunc_data_mcf_abs_mean.rms sfunc_data_mcf_rel.rms sfunc_data_mcf_rel_mean.rms mc
flirt -in sfunc_data_mcf -ref /nfs/diskstation/projects/emotionPro/nii/ZZZZ/3danat/reg_fsl/T1_brain.nii.gz -init example_func2highres.mat -applyxfm -out asfunc_data_mcf
applywarp -i asfunc_data_mcf -o mni_sfunc_data_mcf -r /nfs/cell_b/software/fsl/data/standard/MNI152_T1_2mm_brain -w /nfs/diskstation/projects/emotionPro/nii/ZZZZ/3danat/reg_fsl/highres2standard_warp_2mm.nii.gz
rm asfunc_data_mcf.nii.gz
fslmaths mni_sfunc_data_mcf -bptf 25 -1 mni_sfunc_data_mcf_hp
