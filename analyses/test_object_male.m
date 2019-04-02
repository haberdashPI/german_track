run('../util/setup.m')

% First sanity check: can we more accurately recover the male voice
% when that is the voice listeners are asked to attend to.

% all correlations are the same: are they all working "well", let's try
% a randomly selected sound?

stim_info = read_json(fullfile(stimulus_dir,'config.json'));
eeg_files = dir(fullfile(data_dir,'eeg_response*.mat'));
use_fake_data = 0;

male_C = [];
fem1_C = [];
fem2_C = [];
other_male_C = [];

sid = {};
for i = 1:length(eeg_files)
  [eeg,stim_events,cur_sid] = load_subject(eeg_files(i).name);

  maxlag = 0.25;

  male_index = 1;
  fem1_index = 2;
  fem2_index = 3;

  lags = 0:round(maxlag*eeg.fsample);

  if use_fake_data
    use_eeg1 = trf_fake_data(eeg,stim_info,3,1,20,...
      @(i)strcmp(stim_events{i,'condition'},'object'),...
      @(i)load_sentence(stim_events,stim_info,i,male_index));
    use_eeg2 = trf_fake_data(eeg,stim_info,3,21,25,...
      @(i)strcmp(stim_events{i,'condition'},'object'),...
      @(i)load_sentence(stim_events,stim_info,i,fem1_index));
    use_eeg3 = trf_fake_data(eeg,stim_info,3,31,35,...
      @(i)strcmp(stim_events{i,'condition'},'object'),...
      @(i)load_sentence(stim_events,stim_info,i,fem2_index));

    use_eeg = [];
    use_eeg.fsample = use_eeg1.fsample;
    use_eeg.trial = cell(size(eeg.trial));
    for i = 1:length(use_eeg.trial)
      if length(use_eeg1.trial{i}) > 0
        use_eeg.trial{i} = use_eeg1.trial{i} + use_eeg2.trial{i} + ...
          use_eeg3.trial{i};
      end
    end
  else
    use_eeg = eeg;
  end

  male_model = trf_train(use_eeg,stim_info,lags,...
    @(i)strcmp(stim_events{i,'condition'},'object'),...
    @(i)load_sentence(stim_events,stim_info,i,male_index));

  fem1_model = trf_train(use_eeg,stim_info,lags,...
    @(i)strcmp(stim_events{i,'condition'},'object'),...
    @(i)load_sentence(stim_events,stim_info,i,fem1_index));

  fem2_model = trf_train(use_eeg,stim_info,lags,...
    @(i)strcmp(stim_events{i,'condition'},'object'),...
    @(i)load_sentence(stim_events,stim_info,i,fem2_index));

  other_male_model = trf_train(use_eeg,stim_info,lags,...
    @(i)strcmp(stim_events{i,'condition'},'object'),...
    @(i)load_other_sentence(stim_events,stim_info,i,male_index));

  this_male_C = trf_corr(use_eeg,stim_info,male_model,lags,...
    @(i)strcmp(stim_events{i,'condition'},'object'),...
    @(i)load_sentence(stim_events,stim_info,i,male_index))';

  this_fem1_C = trf_corr(use_eeg,stim_info,fem1_model,lags,...
    @(i)strcmp(stim_events{i,'condition'},'object'),...
    @(i)load_sentence(stim_events,stim_info,i,fem1_index))';

  this_fem2_C = trf_corr(use_eeg,stim_info,fem2_model,lags,...
    @(i)strcmp(stim_events{i,'condition'},'object'),...
    @(i)load_sentence(stim_events,stim_info,i,fem2_index))';

  this_other_male_C = trf_corr(use_eeg,stim_info,other_male_model,lags,...
    @(i)strcmp(stim_events{i,'condition'},'object'),...
    @(i)load_sentence(stim_events,stim_info,i,male_index))';

  cur_sids = cell(length(this_fem1_C),1);
  cur_sids(:) = {cur_sid};

  male_C = [male_C; this_male_C];
  fem1_C = [fem1_C; this_fem1_C];
  fem2_C = [fem2_C; this_fem2_C];
  other_male_C = [other_male_C; this_other_male_C];
  sid = [sid; cur_sids];
end

if use_fake_data
  writetable(table(male_C,fem1_C,fem2_C,other_male_C,sid),fullfile(cache_dir,'fake_testobj.csv'));
else
  writetable(table(male_C,fem1_C,fem2_C,other_male_C,sid),fullfile(cache_dir,'testobj.csv'));
end
alert()
