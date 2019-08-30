
# Plans

## checks
+ verify `test_correct`

+ analyze behavioral data

## new features
- filter eeg data using 
  - let's start with something simple: include these features of the sound
    + N-bank logarithmic filtering of the sound (N = 3-4)
    + pitch derivative
    + verify that the pitch outputs from CEDAR are reasonable
    + test pitch in analysis (seems to work better, maybe step above not needed, worth looking at though)

  + TODO: reorganize analysis files into a local package
  + include these features of the EEG
    + eeg channels
    + alpha band amplitudes and phases (5-15Hz) in log-frequency bands?
    + gamma band amplitudes and phases (30-100Hz) in log-frequency bands?
    + test alpha in analysis
    + test gamma in analysis

## new analysis config
- do we compare across conditions (with same stimulus and subject)
    - does the target event get detected better across conditions?
    - in online decoding, does the increase look different?
- do we compare across times (e.g. decoding better near target)
  + does decoding differ by target locus
  - does decoding differ near switches

## verifications
    - check out the feature weightings w.r.t the stimulus features
      and the eeg features, plot them

# algorithm refinement
- try an L1 loss
  + implement
  - run on cluster
+ try L1 loss instead of cor
- cross validate regularization parameter
- try a 1-layer (should be similar to regression), and then multilayer DNN
