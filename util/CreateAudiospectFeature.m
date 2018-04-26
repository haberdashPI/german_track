function spect = CreateAudiospectFeature(x,fs,efs)
  full_spect = wav2aud(resample(x,8000,fs),[8 4 0.1*sqrt(mean(x.^2)) -1]);
  spect = resample(full_spect,efs,1000/8);
end
