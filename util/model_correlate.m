function cor = model_correlate(eeg_data,efs,envelope,config,model,weights)
  c = corrcoef(model_cor_data(eeg_data,efs,envelope,config,model,weights))
  cor = c(1,2);
end
