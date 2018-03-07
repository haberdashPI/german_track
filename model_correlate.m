function cor = model_correlate(eeg_data,efs,envelope,config,model,trf)
  c = corrcoef(model_cor_data(eeg_data,efs,envelope,config,model,trf))
  cor = c(1,2);
end
