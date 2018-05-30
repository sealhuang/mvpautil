function designmat = mkdesign(subj, ridx)
% 
% make design matrix from behavioral record.
% designmat = mkdesign(subj, ridx)
%     subj: subject name
%     ridx: run index

% load behavior record info
root_dir = '/nfs/diskstation/projects/emotionPro';
beh_dir = fullfile(root_dir, 'beh', 'mat');
mat_name =  strcat('final_',subj,'_record_run_',num2str(ridx),'.txt.mat');
record = load(fullfile(beh_dir, mat_name));
labels = cell2mat(record.fresult(:, 6));

% init design matrix
designmat = zeros(355, 4);
for i=1:length(labels)
    designmat(4*i, labels(i)) = 1;
end

end
