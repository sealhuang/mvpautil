function [design, data, hrfs, beta, r2] = modelestimate_full(sid, session)
% 
% Script for distinct image neural response estimate.
% [design, data, hrfs, beta, r2] = modelestimate_full(sid, session)
%     sid: subject index
%     session: session index

% subject names
subj_names = {'liqing', 'zhangjipeng', 'zhangdan', 'wanghuicui', ...
              'zhuzhiyuan', 'longhailiang', 'liranran'};
subj = subj_names{sid};
% dir config
root_dir = '/nfs/diskstation/projects/emotionPro';
nii_dir = fullfile(root_dir, 'workshop', 'glmmodel', 'nii');
% config run list
run_list = reshape(1:10, 5, 2);
run_list = run_list(:, session);
% design and data cell init
design = cell(1, length(run_list));
data = cell(1, length(run_list));

for i=1:length(run_list)
    design{i} = mkdesign(subj, run_list(i), 1);
    nii_file = fullfile(nii_dir, strcat('S', num2str(sid)), ...
                        strcat('mcsfunc_', num2str(run_list(i)), '.nii.gz'));
    nii = load_nii(nii_file);
    size(nii.img)
    data{i} = nii.img;
end

% experiment parameter config
stimdur = 2;
tr = 2;

% HRF init
hrf = getcanonicalhrf(stimdur, tr)';

% Define some useful constants.
xyzsize = [64 64 33];  % the XYZ dimensions of the dataset
numcond = 80;         % the number of conditions in this dataset

% Define an options struct.  This specifies (1) that the HRF estimated 
% from the data should always be used (even if it is very unlike the initial seed),
% and (2) that we want to suppress text output (since many voxels will be analyzed
% in a loop).
opt = struct('hrfthresh', -Inf, 'suppressoutput', 1);

% Initialize the results.
hrfs = zeros([xyzsize length(hrf)], 'single');
beta = zeros([xyzsize numcond*5], 'single');
r2 = zeros([xyzsize], 'single');

% Loop over voxels.  Note: The following loop may take a long time to execute.
% This loop can be speeded up using parfor or cluster-computing solutions, but
% here we use a simple for-loop for the purpose of simplicity.
cache0 = [];
for zz=1:xyzsize(3)
    fprintf('\nslice %d #', zz);
    for xx=1:xyzsize(1)
        fprintf('.');
        for yy=1:xyzsize(2)
            % Extract the time-series data for one voxel.
            data0 = cellfun(@(x) flatten(x(xx,yy,zz,:)), data, 'UniformOutput', 0);
            % We use the canonical HRF as the initial seed for the HRF.
            [results0, cache0] = GLMmodelestimate(design, data0, stimdur, tr, ...
                                                  'optimize', hrf, opt, cache0);
            % Record the results.
            hrfs(xx, yy, zz, :) = results0.modelmd{1};
            beta(xx, yy, zz, :) = results0.modelmd{2};
            r2(xx, yy, zz) = results0.R2;
        end
    end
end

hrfs = flip(hrfs, 1);
beta = flip(beta, 1);
r2 = flip(r2, 1);

save(strcat('S', num2str(sid), '_results_s', num2str(session), '_full.mat'), 'hrfs', 'beta', 'r2', 'design', 'data')

end

