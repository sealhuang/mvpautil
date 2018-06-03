function designmat = mkdesign(subj, run_idx, condition)
% 
% make design matrix from behavioral record.
% designmat = mkdesign(subj, run_idx, condition)
%     subj: subject name
%     run_idx: run index
%     condition: experimental condition flag, 0 for emotion condition, 1
%     for distinct image condition

% load behavior record info
%root_dir = '/nfs/diskstation/projects/emotionPro';
%beh_dir = fullfile(root_dir, 'beh', 'mat');
root_dir = '/Users/sealhuang/project/emotionPro';
beh_dir = fullfile(root_dir, 'beh');
mat_name =  strcat('final_', subj, '_record_run_', ...
                   num2str(run_idx), '.txt.mat');
record = load(fullfile(beh_dir, mat_name));

if condition
    % for distinct image condition
    designmat = zeros(355, 80);
    distinct_imgs = cell(80, 1);
    for i=1:length(record.fresult)
        j = 1;
        while ~strcmp(record.fresult{i, 3}, distinct_imgs{j}) && ~isempty(distinct_imgs{j})
            j = j + 1;
        end
        if isempty(distinct_imgs{j})
            distinct_imgs{j} = record.fresult{i, 3};
        end
        designmat(4*i, j) = 1;
    end 
else
    % for emotion condition
    designmat = zeros(355, 4);
    labels = cell2mat(record.fresult(:, 6));    
    for i=1:length(labels)
        designmat(4*i, labels(i)) = 1;
    end
end
