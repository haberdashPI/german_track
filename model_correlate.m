function cor = model_correlate(start,stop,eeg_data,model,trf,kind)
  [~,prediction] = FindTRF([],[],-1,eeg_data(:,start:stop)',trf.(kind),...
                           model.lags,'Shrinkage');
  envelope = model.envelope.(kind);
  c = corrcoef([prediction envelope(start:stop)]);
  cor = c(1,2);
end
