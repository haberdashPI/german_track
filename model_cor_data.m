function cor_data = model_cor_data(eeg_data,efs,envelope,config,model,trf)
  start = max(1,floor(efs * config.start));
  stop = min(ceil(efs * config.stop),size(eeg_data,2));

  [~,prediction] = FindTRF([],[],-1,eeg_data(:,start:stop)',trf,...
                           model.lags,'Shrinkage');
  if length(envelope) < stop
    envelope(end+1:stop) = 0.0;
  end

  cor_data = [prediction envelope(start:stop,:)];
end
