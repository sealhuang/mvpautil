function [design, data, s1results, hrfs, betas, R2] = glmdenoise_pipeline(subj, session)
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

% experiment parameter config
stimdur = 2;
tr = 2;

% A 2-stage analyse pipeine for extimate voxel-specific HRF
% Step 1: Initial call to GLMdenoise to learn the noise regressors
% Compute a canonical HRF.
hrf = getcanonicalhrf(stimdur, tr)';
% Make the initial call to GLMdenoise.  We indicate that the canonical HRF is to be 
% assumed.  We also turn off bootstrapping as it is unnecessary here.
s1results = GLMdenoisedata(design, data, stimdur, tr, ...
            'assume', hrf, struct('numboots',0), 'stage1figures');

% Extract the noise regressors based on the results.  Note that results.pcnum is 
% the number of noise regressors that was selected by GLMdenoisedata.m.
noisereg = cellfun(@(x) x(:,1:s1results.pcnum), s1results.pcregressors,'UniformOutput',0);
% Inspect the dimensionality of the noise regressors.
noisereg

% Step 2: Re-analyze the data, tailoring the HRF voxel-by-voxel
% Define some useful constants.
xyzsize = [64 64 33];  % the XYZ dimensions of the dataset
numcond = 4;           % the number of conditions in this dataset

% Define an options struct.  This specifies (1) that the noise regressors determined
% above will serve as extra regressors in the GLM, (2) that the HRF estimated 
% from the data should always be used (even if it is very unlike the initial seed),
% and (3) that we want to suppress text output (since many voxels will be analyzed
% in a loop).
opt = struct('extraregressors', {noisereg}, 'hrfthresh', -Inf, ...
             'suppressoutput', 1);

% Initialize the results.
hrfs = zeros([xyzsize length(hrf)], 'single');
betas = zeros([xyzsize numcond], 'single');
R2 = zeros([xyzsize], 'single');

% Loop over voxels.  Note: The following loop may take a long time to execute.
% This loop can be speeded up using parfor or cluster-computing solutions, but
% here we use a simple for-loop for the purpose of simplicity.
cache = [];
for zz=1:xyzsize(3)
    fprintf('\nslice %d #', zz);
    for xx=1:xyzsize(1)
        fprintf('.');
        for yy=1:xyzsize(2)
            % Extract the time-series data for one voxel.
            data0 = cellfun(@(x) flatten(x(xx,yy,zz,:)), data, 'UniformOutput', 0);

            % Analyze the time-series, specifying that we want to optimize the HRF.
            % We use the canonical HRF as the initial seed for the HRF.
            [results0, cache] = GLMestimatemodel(design, data0, stimdur, tr, ...
                                                'optimize', hrf, 0, opt, cache);

            % Record the results.
            hrfs(xx, yy, zz, :) = results0.modelmd{1};
            betas(xx, yy, zz, :) = results0.modelmd{2};
            R2(xx, yy, zz) = results0.R2;
        end
    end
end

%[results, denoiseddata] = GLMdenoisedata(design, data, stimdur, tr, [], [], [], 'figures');
end

