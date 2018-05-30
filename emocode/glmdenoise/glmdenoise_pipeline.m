function [design, data, results, denoiseddata] = glmdenoise_pipeline(subj, session)
% 
% GLMDenoise pipeline script
% [results, denoiseddata] = glmdenoise(subj, session)
%     subj: subject name
%     session: session index

% dir config
root_dir = '/nfs/diskstation/projects/emotionPro';
nii_dir = fullfile(root_dir, 'workshop', 'glmdenoise', 'nii');
% config run list
run_list = reshape(1:10, 5, 2);
run_list = run_list(:, session);
% design and data cell init
design = cell(1, length(run_list));
data = cell(1, length(run_list));

for i=1:length(run_list)
    rundesign = mkdesign(subj, run_list(i));
    design{i} = rundesign;
    nii_file = fullfile(nii_dir, 'S1', strcat('mcsfunc_', num2str(run_list(i)), '.nii.gz'));
    nii = load_nii(nii_file);
    size(nii.img)
    data{i} = nii.img;
end

stimdur = 2;
tr = 2;
[results, denoiseddata] = GLMdenoisedata(design, data, stimdur, tr, [], [], [], 'figures');
end
    
