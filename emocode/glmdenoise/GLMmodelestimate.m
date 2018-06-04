function [results,cache] = GLMmodelestimate(design,data,stimdur,tr,hrfmodel,hrfknobs,opt,cache,mode)

% function [results,cache] = GLMmodelestimate(design,data,stimdur,tr,hrfmodel,hrfknobs,opt,cache)
%
% <design> is the experimental design.  There are three possible cases:
%   1. A where A is a matrix with dimensions time x conditions.
%      Each column should be zeros except for ones indicating condition onsets.
%      (Fractional values in the design matrix are also allowed.)
%   2. {A1 A2 A3 ...} where each of the A's are like the previous case.
%      The different A's correspond to different runs, and different runs
%      can have different numbers of time points.
% <data> is the time-series data with dimensions X x Y x Z x time or a cell vector of 
%   elements that are each X x Y x Z x time.  XYZ can be collapsed such that the data 
%   are given as a 2D matrix (XYZ x time); however, if you do this, then several of the 
%   figures that are written out by this function will not be useful to look at.  The 
%   dimensions of <data> should mirror that of <design>.  (For example, <design> and 
%   <data> should have the same number of runs, the same number of time points, etc.)
%   <data> should not contain any NaNs. We automatically convert <data> to 
%   single format (if necessary).
% <stimdur> is the duration of a trial in seconds
% <tr> is the sampling rate in seconds
% <hrfmodel> indicates the type of model to use for the HRF:
%   'fir' indicates a finite impulse response model (a separate timecourse
%     is estimated for every voxel and every condition)
%   'assume' indicates that the HRF is provided (see <hrfknobs>)
%   'optimize' indicates that we should estimate a global HRF from the data
% <hrfknobs> (optional) is as follows:
%   if <hrfmodel> is 'fir', then <hrfknobs> should be the number of 
%     time points in the future to model (N >= 0).  For example, if N is 10, 
%     then timecourses will consist of 11 points, with the first point 
%     coinciding with condition onset.
%   if <hrfmodel> is 'assume', then <hrfknobs> should be time x 1 with
%     the HRF to assume.
%   if <hrfmodel> is 'optimize', then <hrfknobs> should be time x 1 with the 
%     initial seed for the HRF.  The length of this vector indicates the
%     number of time points that we will attempt to estimate in the HRF.
%   Note on normalization:  In the case that <hrfmodel> is 'assume' or
%   'optimize', we automatically divide <hrfknobs> by the maximum value
%   so that the peak is equal to 1.  And if <hrfmodel> is 'optimize',
%   then after fitting the HRF, we again normalize the HRF to peak at 1
%   (and adjust amplitudes accordingly).  Default in the case of 'fir' is
%   20.  Default in the case of 'assume' and 'optimize' is to use a 
%   canonical HRF that is calculated based on <stimdur> and <tr>.
% <opt> (optional) is a struct with the following fields:
%   <extraregressors> (optional) is time x regressors or a cell vector
%     of elements that are each time x regressors.  The dimensions of 
%     <extraregressors> should mirror that of <design> (i.e. same number of 
%     runs, same number of time points).  The number of extra regressors 
%     does not have to be the same across runs, and each run can have zero 
%     or more extra regressors.  If [] or not supplied, we do 
%     not use extra regressors in the model.
%   <maxpolydeg> (optional) is a non-negative integer with the maximum 
%     polynomial degree to use for polynomial nuisance functions, which
%     are used to capture low-frequency noise fluctuations in each run.
%     Can be a vector with length equal to the number of runs (this
%     allows you to specify different degrees for different runs).  
%     Default is to use round(L/2) for each run where L is the 
%     duration in minutes of a given run.
%   <seed> (optional) is the random number seed to use (this affects
%     the selection of bootstrap samples). Default: sum(100*clock).
%   <numforhrf> (optional) is a positive integer indicating the number 
%     of voxels (with the best R^2 values) to consider in fitting the 
%     global HRF.  This input matters only when <hrfmodel> is 'optimize'.
%     Default: 50.  (If there are fewer than that number of voxels
%     available, we just use the voxels that are available.)
%   <hrffitmask> (optional) is X x Y x Z with 1s indicating all possible
%     voxels to consider for fitting the global HRF.  This input matters
%     only when <hrfmodel> is 'optimize'.  Special case is 1 which means
%     all voxels can be potentially chosen.  Default: 1.
%   <wantpercentbold> (optional) is whether to convert the amplitude estimates
%     in 'models', 'modelmd', and 'modelse' to percent BOLD change.  This is
%     done as the very last step, and is accomplished by dividing by the 
%     absolute value of 'meanvol' and multiplying by 100.  (The absolute 
%     value prevents negative values in 'meanvol' from flipping the sign.)
%     Default: 1.
%   <hrfthresh> (optional) is an R^2 threshold.  If the R^2 between the estimated
%      HRF and the initial HRF is less than <hrfthresh>, we decide to just use
%      the initial HRF.  Set <hrfthresh> to -Inf if you never want to reject
%      the estimated HRF.  Default: 50.
%   <suppressoutput> (optional) is whether to suppress fprintf statements.  Default: 0.
% <cache> (optional) is used for speeding up execution.  If you are calling this
%   function with identical inputs except potentially for different <data>, then
%   if you can take the <cache> returned by the first call and re-use it for
%   subsequent calls.
%
% Based on the experimental design (<design>, <stimdur>, <tr>) and the model 
% specification (<hrfmodel>, <hrfknobs>), fit a GLM model to the data (<data>) 
% using a certain resampling scheme (<resampling>).
%
% Return <results> as a struct containing the following fields:
% <models> contains the full set of model estimates (e.g. all bootstrap results)
% <modelmd> contains the final model estimate (median of the estimates in <models>)
% <modelse> contains the standard error of the final model estimate (half of the
%   68% range of the estimates in <models>).  Note that <modelse> will be
%   computed in all resampling schemes (full-fit, bootstrapping, and 
%   cross-validation) but can be interpreted as an estimate of standard 
%   error only in the bootstrapping scheme.
% <R2> is X x Y x Z with model accuracy expressed in terms of R^2 (percentage).
%   In the full-fit and bootstrap cases, <R2> is an R^2 value indicating how
%     well the final model estimate (<modelmd>) fits the data.
%   In the cross-validation case, <R2> is an R^2 value indicating how well
%     the cross-validated predictions of the model match the data.  (The
%     predictions and the data are each aggregated across runs before
%     the computation of R^2.)
% <R2run> is X x Y x Z x runs with R^2 values calculated on a per-run basis.
% <signal> is X x Y x Z with the maximum absolute amplitude in <modelmd>
%   (this is computed over all conditions and time points in the case of 'fir'
%   and over all conditions in the case of 'assume' and 'optimize').
% <noise> is X x Y x Z with the average amplitude error in <modelse>.
% <SNR> is X x Y x Z with <signal> divided by <noise>.
% <hrffitvoxels> is X x Y x Z with 1s indicating the voxels used for fitting
%   the global HRF.  This input is returned as [] if <hrfmodel> is not 'optimize'.
%   In the bootstrap and cross-validation cases, <hrffitvoxels> indicates the
%   voxels corresponding to the last iteration.
% <meanvol> is X x Y x Z with the mean of all volumes
% <inputs> is a struct containing all inputs used in the call to this
%   function, excluding <data>.  We additionally include a field called 
%   'datasize' which contains the size of each element of <data>.
%
% Additional details on the format of <models>, <modelmd>, and <modelse>:
% - If <hrfmodel> is 'fir', then model estimates consist of timecourses:
%   <models> is X x Y x Z x conditions x time x resamples
%   <modelmd> is X x Y x Z x conditions x time
%   <modelse> is X x Y x Z x conditions x time
% - If <hrfmodel> is 'assume' or 'optimize', then model estimates consist
%     of HRF estimates and amplitude estimates:
%   <models> is {A B} where A is time x resamples (HRF estimates)
%     and B is X x Y x Z x conditions x resamples (amplitude estimates)
%   <modelmd> is {A B} where A is time x 1 and B is X x Y x Z x conditions
%   <modelse> is {A B} where A is time x 1 and B is X x Y x Z x conditions
%
% Notes on model accuracy (R^2):
% - We quantify the accuracy of the GLM model as the amount of variance in the
% time-series data that is explained by the deterministic portion of the model,
% that is, the hemodynamic responses evoked by the various experimental conditions.
% Note that this does not include the nuisance components of the model, that is, 
% the polynomial regressors and any extra regressors provided by the user
% (see opt.extraregressors).
% - The metric that we use for accuracy is R^2.  Specifically:
%     R^2 = 100 * (1-sum((data-model)^2)/sum(data^2))
% - Before computing R^2 between the model and the data, we project out
% polynomial regressors from both the model and the data. The purpose of 
% this is to reduce the influence of low-frequency fluctuations (which
% can be quite large in fMRI data) on the model accuracy metric.
%
% Notes on the estimation of a global HRF:
% - When <hrfmodel> is 'optimize', we estimate a global HRF from the data.  
% This is achieved using an iterative fitting strategy:  First, the HRF is fixed 
% to the initial seed provided by <hrfknobs>, and we estimate the amplitudes
% using OLS.  Then, the amplitudes are fixed (to the estimates obtained in
% the previous step), and we estimate the HRF using OLS.  Next, the HRF is fixed
% (to the estimate obtained in the previous step), and we re-estimate the
% amplitudes using OLS.  This process is repeated until convergence.
% - The reason for the iterative fitting strategy is that the entire model 
% cannot be estimated at once using linear fitting techniques (and nonlinear
% techniques would be too costly).
% - At the HRF-estimation steps of the fitting process, the entire dataset can 
% in theory be fit.  However, this is undesirable for two reasons.  One, 
% fitting the entire dataset may have exorbitant memory requirements.  
% Two, assuming that most voxels are unrelated to the experimental paradigm 
% (as is typically the case in an fMRI experiment), fitting the entire dataset
% will result in a poor-quality (noisy) HRF.  To resolve these issues, we use 
% a strategy in which we determine the best voxels in terms of R^2 at a given 
% amplitude-estimation step and fit only these voxels in the subsequent
% HRF-estimation step.  The number of voxels that are chosen is controlled 
% by opt.numforhrf, and the pool of chosen voxels is updated at each
% amplitude-estimation step.
% - In some cases, the fitted global HRF may diverge wildly from the initial 
% seed.  This may indicate extremely low SNR and/or a problem with the coding
% of the experimental design and/or a poor initial seed for the HRF.  If the
% R^2 between the initial seed and the fitted global HRF is less than opt.hrfthresh,
% we issue a warning and simply use the initial seed as the HRF (instead of
% relying on the fitted global HRF).  These cases should be inspected and
% troubleshooted on a case-by-case basis.  (In GLMdenoisedata.m, a figure
% named "HRF.png" is created --- if the initial and estimated HRF are 
% exactly overlapping on the figure, this indicates that the exception 
% case occured.)
%
% Additional information:
% - In some circumstances (e.g. using a FIR model with insufficient data),
% the design matrix may be singular and there is no unique solution.  Our
% strategy for these cases is as follows: If MATLAB issues a warning during 
% the inversion of the autocorrelation matrix (i.e. X'*X), then program 
% execution halts.
%
% Internal input:
% <mode> (optional) is
%   1 means that only the 'R2' output is desired (to save computation time)
%   2 means that hrfmodel is 'optimize', resampling is 0, and we only care
%     about the hrf and hrffitvoxels outputs (to save time and memory)
%   Default: 0.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DEAL WITH INPUTS, ETC.

% input
if ~exist('hrfknobs','var') || isempty(hrfknobs)
  if isequal(hrfmodel,'fir')
    hrfknobs = 20;
  else
    hrfknobs = normalizemax(getcanonicalhrf(stimdur,tr)');
  end
end
if ~exist('opt','var') || isempty(opt)
  opt = struct();
end
if ~exist('cache','var') || isempty(cache)
  cache = [];
end
if ~exist('mode','var') || isempty(mode)
  mode = 0;
end

% massage input
if ~iscell(design)
  design = {design};
end
if ~iscell(data)
  data = {data};
end
for p=1:length(data)
  if ~isa(data{p},'single')
    data{p} = single(data{p});
  end
end

% calc
numruns = length(design);
is3d = size(data{1},4) > 1;
if is3d
  dimdata = 3;
  dimtime = 4;
  xyzsize = sizefull(data{1},3);
else
  dimdata = 1;
  dimtime = 2;
  xyzsize = size(data{1},1);
end
numvoxels = prod(xyzsize);

% deal with defaults
if ~isfield(opt,'extraregressors') || isempty(opt.extraregressors)
  opt.extraregressors = cell(1,numruns);
end
if ~isfield(opt,'maxpolydeg') || isempty(opt.maxpolydeg)
  opt.maxpolydeg = zeros(1,numruns);
  for p=1:numruns
    opt.maxpolydeg(p) = round(((size(data{p},dimtime)*tr)/60)/2);
  end
end
if ~isfield(opt,'seed') || isempty(opt.seed)
  opt.seed = sum(100*clock);
end
if ~isfield(opt,'numforhrf') || isempty(opt.numforhrf)
  opt.numforhrf = 50;
end
if ~isfield(opt,'hrffitmask') || isempty(opt.hrffitmask)
  opt.hrffitmask = 1;
end
if ~isfield(opt,'wantpercentbold') || isempty(opt.wantpercentbold)
  opt.wantpercentbold = 1;
end
if ~isfield(opt,'hrfthresh') || isempty(opt.hrfthresh)
  opt.hrfthresh = 50;
end
if ~isfield(opt,'suppressoutput') || isempty(opt.suppressoutput)
  opt.suppressoutput = 0;
end
if isequal(hrfmodel,'assume') || isequal(hrfmodel,'optimize')
  hrfknobs = normalizemax(hrfknobs);
end
if length(opt.maxpolydeg) == 1
  opt.maxpolydeg = repmat(opt.maxpolydeg,[1 numruns]);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CALCULATE MEAN VOLUME

volcnt = cellfun(@(x) size(x,dimtime),data);
temp = {};  % using this instead of cellfun to reduce memory usage
for p=1:length(data)
  temp{p} = squish(mean(data{p},dimtime),dimdata);
end
meanvol = reshape(catcell(2,temp) * (volcnt' / sum(volcnt)),[xyzsize 1]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DEAL WITH NUISANCE COMPONENTS

% construct projection matrices for the nuisance components
polymatrix = {};
exmatrix = {};
combinedmatrix = {};
for p=1:numruns

  % this projects out polynomials
  pmatrix = constructpolynomialmatrix(size(data{p},dimtime),0:opt.maxpolydeg(p));
  polymatrix{p} = projectionmatrix(pmatrix);

  % this projects out the extra regressors
  if isempty(opt.extraregressors{p})
    exmatrix{p} = 1;
  else
    exmatrix{p} = projectionmatrix(opt.extraregressors{p});
  end
  
  % this projects out both of them
  combinedmatrix{p} = projectionmatrix(cat(2,pmatrix,opt.extraregressors{p}));

end

% project out nuisance components from the data.
% after this step, data will have polynomials removed,
% and data2 will have both polynomials and extra regressors removed.
data2 = {};  % NOTE: data and data2 are big --- be careful of MEMORY usage.
for p=1:numruns
  data{p} = squish(data{p},dimdata)';
  data2{p} = combinedmatrix{p}*data{p};
  data{p} = polymatrix{p}*data{p};
end
% note that data and data2 are now in flattened format (time x voxels)!!

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% FIT MODELS

% fit the model to the entire dataset.  we obtain just one analysis result.
if ~opt.suppressoutput, fprintf('fitting model...'); end
results = {};
[results{1}, hrffitvoxels, cache] = fitmodel_helper(design, data2, tr, ...
        hrfmodel, hrfknobs, opt, combinedmatrix, dimdata, dimtime, cache);
if ~opt.suppressoutput, fprintf('done.\n'); end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PREPARE MODEL ESTIMATES FOR OUTPUT

if ~opt.suppressoutput, fprintf('preparing output...'); end
switch hrfmodel

case 'fir'

  results = struct('models',cat(4,results{:}));  % voxels x conditions x time x resamples
  if size(results.models,4) == 1
    results.modelmd = results.models;
    results.modelse = zeros(size(results.models),class(results.models));
  else
    temp = zeros([sizefull(results.models,3) 3],class(results.models));
    for p=1:size(results.models,3)  % ugly to avoid memory usage
      temp(:,:,p,:) = prctile(results.models(:,:,p,:),[16 50 84],4);
    end
    results.modelmd = temp(:,:,:,2);
    results.modelse = diff(temp(:,:,:,[1 3]),1,4)/2;
    clear temp;
  end

  % massage format
  sz = sizefull(results.models,4);
  results.models = reshape(results.models,[xyzsize sz(2:4)]);
  results.modelmd = reshape(results.modelmd,[xyzsize sz(2:3)]);
  results.modelse = reshape(results.modelse,[xyzsize sz(2:3)]);

case {'assume' 'optimize'}

  temp = catcell(2,cellfun(@(x) x(1),results));
  results = catcell(3,cellfun(@(x) x(2),results));  % beware of MEMORY here
  results = struct('models',{{temp results}});  % ugly to avoid memory usage

  % deal with {1}
  if size(results.models{1},2) == 1
    results.modelmd{1} = results.models{1};
    results.modelse{1} = zeros(size(results.models{1}),class(results.models{1}));
  else
    temp = prctile(results.models{1},[16 50 84],2);
    results.modelmd{1} = temp(:,2);
    results.modelse{1} = diff(temp(:,[1 3]),1,2)/2;
  end

  % deal with {2}
  if size(results.models{2},3) == 1
    results.modelmd{2} = results.models{2};
    results.modelse{2} = zeros(size(results.models{2}),class(results.models{2}));
  else
    temp = zeros([sizefull(results.models{2},2) 3],class(results.models{2}));
    for p=1:size(results.models{2},2)  % ugly to avoid memory usage
      temp(:,p,:) = prctile(results.models{2}(:,p,:),[16 50 84],3);
    end
    results.modelmd{2} = temp(:,:,2);
    results.modelse{2} = diff(temp(:,:,[1 3]),1,3)/2;
    clear temp;
  end

  % massage format
  sz = sizefull(results.models{2},3);
  results.models{2} = reshape(results.models{2},[xyzsize sz(2:3)]);
  results.modelmd{2} = reshape(results.modelmd{2},[xyzsize sz(2)]);
  results.modelse{2} = reshape(results.modelse{2},[xyzsize sz(2)]);

end
if ~opt.suppressoutput, fprintf('done.\n'); end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% COMPUTE MODEL FITS

if ~(mode==2)
  if ~opt.suppressoutput, fprintf('computing model fits...'); end
  % compute the time-series fit corresponding to the final model estimate
  modelfit = GLMpredict(results.modelmd,design,tr,cellfun(@(x) size(x,1),data),dimdata);
  if ~opt.suppressoutput, fprintf('done.\n'); end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% COMPUTE R^2

if ~(mode==2 || mode==3)

  if ~opt.suppressoutput, fprintf('computing R^2...'); end

  % remove polynomials from the model fits (or predictions)
  modelfit = cellfun(@(a,b) a*squish(b,dimdata)',polymatrix,modelfit,'UniformOutput',0);  % result is time x voxels

  % calculate overall R^2 [beware: MEMORY]
  results.R2 = reshape(calccodcell(modelfit,data,1)',[xyzsize 1]);  % notice that we use 'data' not 'data2'

  % calculate R^2 on a per-run basis [beware: MEMORY]
  results.R2run = catcell(dimdata+1,cellfun(@(x,y) reshape(calccod(x,y,1,0,0)',[xyzsize 1]),modelfit,data,'UniformOutput',0));

  % clear
  clear modelfit;  % big memory usage

  if ~opt.suppressoutput, fprintf('done.\n'); end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% COMPUTE SNR

if ~opt.suppressoutput, fprintf('computing SNR...'); end

if ~(mode==2)

  switch hrfmodel
  
  case 'fir'
    results.signal = max(max(abs(results.modelmd),[],dimdata+1),[],dimdata+2);
    results.noise = mean(mean(results.modelse,dimdata+1),dimdata+2);
    results.SNR = results.signal ./ results.noise;
  
  case {'assume' 'optimize'}
    results.signal = max(abs(results.modelmd{2}),[],dimdata+1);
    results.noise = mean(results.modelse{2},dimdata+1);
    results.SNR = results.signal ./ results.noise;
    
  end

end

if ~opt.suppressoutput, fprintf('done.\n'); end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PREPARE ADDITIONAL OUTPUTS

% this is a special case
if isempty(hrffitvoxels)
  results.hrffitvoxels = [];
else
  results.hrffitvoxels = copymatrix(zeros([xyzsize 1]),hrffitvoxels,1);
end
results.meanvol = meanvol;

% return all the inputs (except for the data) in the output.
% also, include a new field 'datasize'.
results.inputs.design = design;
results.inputs.datasize = cellfun(@(x) size(x),data,'UniformOutput',0);
results.inputs.stimdur = stimdur;
results.inputs.tr = tr;
results.inputs.hrfmodel = hrfmodel;
results.inputs.hrfknobs = hrfknobs;
results.inputs.opt = opt;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CONVERT TO % BOLD CHANGE

if opt.wantpercentbold
  con = 1./abs(results.meanvol) * 100;
  switch hrfmodel
  case 'fir'
    results.models = bsxfun(@times,results.models,con);
    results.modelmd = bsxfun(@times,results.modelmd,con);
    results.modelse = bsxfun(@times,results.modelse,con);
  case {'assume' 'optimize'}
    results.models{2} = bsxfun(@times,results.models{2},con);
    results.modelmd{2} = bsxfun(@times,results.modelmd{2},con);
    results.modelse{2} = bsxfun(@times,results.modelse{2},con);
  end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% HELPER FUNCTION:

function [f,hrffitvoxels,cache] = ...
  fitmodel_helper(design,data2,tr,hrfmodel,hrfknobs,opt,combinedmatrix,dimdata,dimtime,cache)

% if hrfmodel is 'fir', then <f> will be voxels x conditions x time (flattened format)
% if hrfmodel is 'assume' or 'optimize', then <f> will be {A B}
%   where A is time x 1 and B is voxels x conditions (flattened format).
% <hrffitvoxels> is [] unless hrfmodel is 'optimize', in which case it will be
%   a column vector of voxel indices.
%
% note: cache.rawdesign will exist for 'fir' and 'assume' but not 'optimize'.

% internal constants
minR2 = 99;  % in 'optimize' mode, if R^2 between previous HRF and new HRF
             % is above this threshold (and we have at least gone through
             % one complete round of fitting (just so that we can change
             % a little from the initial seed)), then we stop fitting.

% init
hrffitvoxels = [];

switch hrfmodel
case 'fir'
  
  % since 'fir', we can assume design is not the onset case, but check it
  assert(~iscell(design{1}));
  
  % calc
  numconditions = size(design{1},2);
  
  % prepare design matrix
  for p=1:length(design)
    
    % expand original design matrix using delta basis functions.
    % the length of each timecourse is L.
    design{p} = constructstimulusmatrices(full(design{p})',0,hrfknobs,0);  % time x L*conditions
    
    % save a record of the raw design matrix
    cache.rawdesign{p} = design{p};
    
    % remove polynomials and extra regressors
    design{p} = combinedmatrix{p}*design{p};  % time x L*conditions

  end
  
  % merge design
  merged_design = zeros(length(design)*size(design{1}, 1), length(design)*size(design{1}, 2));
  for p=1:length(design)
      merged_design([(1+(p-1)*size(design{1}, 1)):(p*size(design{1}, 1)), ...
                     (1+(p-1)*size(design{1}, 2)):(p*size(design{1}, 2))]) = design{p};
  end
  % fit model
  %f = mtimescell(olsmatrix2(cat(1,design{:})),data2);  % L*conditions x voxels
  f = mtimescell(olsmatrix2(merged_design),data2);  % L*conditions x voxels
  f = permute(reshape(f,hrfknobs+1,numconditions*length(design),[]),[3 2 1]);  % voxels x conditions x L

case 'assume'

  % prepare design matrix
  for p=1:length(design)
      
    % convolve original design matrix with HRF
    ntime = size(design{p},1);                    % number of time points
    design{p} = conv2(full(design{p}),hrfknobs);  % convolve
    design{p} = design{p}(1:ntime,:);             % extract desired subset

    % save a record of the raw design matrix
    cache.rawdesign{p} = design{p};
  
    % remove polynomials and extra regressors
    design{p} = combinedmatrix{p}*design{p};  % time x conditions

  end
  
  % merge design
  merged_design = zeros(length(design)*size(design{1}, 1), length(design)*size(design{1}, 2));
  for p=1:length(design)
      merged_design([(1+(p-1)*size(design{1}, 1)):(p*size(design{1}, 1)), ...
                     (1+(p-1)*size(design{1}, 2)):(p*size(design{1}, 2))]) = design{p};
  end
  
  % fit model
  %f = mtimescell(olsmatrix2(cat(1,design{:})),data2);  % conditions x voxels
  f = mtimescell(olsmatrix2(merged_design),data2);  % conditions x voxels
  f = {hrfknobs f'};

case 'optimize'

  % since 'optimize', we can assume design is not the onset case, but check it
  assert(~iscell(design{1}));

  % calc
  numinhrf = length(hrfknobs);
  numcond = size(design{1},2);
  
  % if cache is empty, fill it
  if isempty(cache)

    % precompute for speed
    design2pre = {};
    for p=1:length(design)
      
      % expand design matrix using delta functions
      ntime = size(design{p},1);              % number of time points
      design2pre{p} = constructstimulusmatrices(full(design{p})',0,numinhrf-1,0);  % time x L*conditions
      design2pre{p} = reshape(design2pre{p},[],numcond);  % time*L x conditions
  
    end
    
    % record it
    cache.design2pre = design2pre;

  % otherwise, use the cache
  else
    design2pre = cache.design2pre;
  end

  % loop until convergence
  currenthrf = hrfknobs;  % initialize
  cnt = 1;
  while 1
  
    % fix the HRF, estimate the amplitudes
    if mod(cnt,2)==1

      % prepare design matrix
      design2 = {};
      for p=1:length(design)
        
        % convolve original design matrix with HRF
        ntime = size(design{p},1);                       % number of time points
        design2{p} = conv2(full(design{p}),currenthrf);  % convolve
        design2{p} = design2{p}(1:ntime,:);              % extract desired subset
        
        % remove polynomials and extra regressors
        design2{p} = combinedmatrix{p}*design2{p};  % time x conditions
    
      end
      
      % merge design
      merged_design = zeros(length(design2)*size(design2{1}, 1), length(design2)*size(design2{1}, 2));
      for p=1:length(design2)
        merged_design((1+(p-1)*size(design2{1}, 1)):(p*size(design2{1}, 1)), ...
                       (1+(p-1)*size(design2{1}, 2)):(p*size(design2{1}, 2))) = design2{p};
      end
      
      % estimate the amplitudes
      %currentbeta = mtimescell(olsmatrix2(cat(1,design2{:})),data2);  % conditions x voxels
      currentbeta = mtimescell(olsmatrix2(merged_design),data2);  % conditions x voxels

      % calculate R^2
      modelfit = merged_design * currentbeta;
      %modelfit = cellfun(@(x) x*currentbeta,design2,'UniformOutput',0);
      modelfit_cell = cell(1, length(design));
      for p=1:length(design)
        modelfit_cell{p} = modelfit((1+(p-1)*355):(p*355));
      end
      R2 = calccodcell(modelfit_cell,data2,1)';
      %R2 = calccodcell(modelfit,data2,1)';
      clear modelfit;
      clear modelfit_cell;
    
      % figure out indices of good voxels
      if isequal(opt.hrffitmask,1)
        temp = R2;
      else
        temp = copymatrix(R2,~opt.hrffitmask(:),-Inf);  % shove -Inf in where invalid
      end
      temp = nanreplace(temp,-Inf);
      [dd,ii] = sort(temp);
      iichosen = ii(max(1,end-opt.numforhrf+1):end);
      iichosen = setdiff(iichosen,iichosen(temp(iichosen)==-Inf));
      hrffitvoxels = iichosen;
   
    % fix the amplitudes, estimate the HRF
    else

      % prepare design matrix
      design2 = {};
      for p=1:length(design)
        
        % calc
        ntime = size(design{p},1);              % number of time points
        
        % weight and sum based on the current amplitude estimates.  only include the good voxels.
        %design2{p} = design2pre{p} * currentbeta(:,hrffitvoxels);  % time*L x voxels
        design2{p} = design2pre{p} * currentbeta((1+(p-1)*80):(p*80),hrffitvoxels);  % time*L x voxels
        
        % remove polynomials and extra regressors
        design2{p} = reshape(design2{p},ntime,[]);  % time x L*voxels
        design2{p} = combinedmatrix{p}*design2{p};  % time x L*voxels
        design2{p} = permute(reshape(design2{p},ntime,numinhrf,[]),[1 3 2]);  % time x voxels x L
    
      end
      
      % estimate the HRF
      previoushrf = currenthrf;
      datasubset = cellfun(@(x) x(:,hrffitvoxels),data2,'UniformOutput',0);
      prev = warning('query'); warning off;
      currenthrf = olsmatrix2(squish(cat(1,design2{:}),2)) * vflatten(cat(1,datasubset{:}));
      warning(prev);

      % if HRF is all zeros (this can happen when the data are all zeros), get out prematurely
      if all(currenthrf==0)
        warning('Script break!');
        break;
      end

      % check for convergence
      if calccod(previoushrf,currenthrf,[],0,0) >= minR2 && cnt > 2
        break;
      end

    end
    
    cnt = cnt + 1;
  
  end
  
  % sanity check
  if calccod(hrfknobs,previoushrf,[],0,0) < opt.hrfthresh
    warning('Global HRF estimate is far from the initial seed, probably indicating low SNR.  We are just going to use the initial seed as the HRF estimate.');
    [f,hrffitvoxels] = fitmodel_helper(design,data2,tr,'assume',hrfknobs,opt,combinedmatrix,dimdata,dimtime,[]);
    return;
  end

  % normalize results
  mx = max(previoushrf);
  previoushrf = previoushrf / mx;
  currentbeta = currentbeta * mx;
  
  % return
  f = {previoushrf currentbeta'};

end
