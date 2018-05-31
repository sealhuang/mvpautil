function save2nii(imgdata, filename)

% dir config
root_dir = '/nfs/diskstation/projects/emotionPro';
nii_dir = fullfile(root_dir, 'workshop', 'glmdenoise', 'nii');
% load template file
template = fullfile(nii_dir, 'S1', 'mcsfunc_1.nii.gz');
template = load_nii(template);

if length(size(imgdata))>3
    template.hdr.dime.dim(5) = size(imgdata, 4);
    template.original.hdr.dime.dim(5) = size(imgdata, 4);
else
    template.hdr.dime.dim(5) = 1;
    template.hdr.dime.dim(5) = 1;
end

template.img = imgdata;
save_nii(template, filename)

