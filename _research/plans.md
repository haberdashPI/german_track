
# Plans

## current steps:

+ contact merve about high v low salience
- break-down freq-bin results by
  - early v late timing
  - high v low salience

  - maybe look at false alarms?

- semi-supervised training, with this approach mounya suggested
  that the loss could be the maximum loss of the three speakers
  at each time point (could also have some sort of continuity cost:
  i.e. transitions in attention are assumed to be more gradual)

- let's try a few more things to see if they help the decoder:
  - train across conditions, with weighting this time
  - try with one feature
- sample combinations of features (many, say 10k) and see if there
  are any combinations that work well. e.g. feature selection
  with a locality constraint

- thorough tests reveal that the index bug was indeed the source of our "success" in decoding. You have to include the tested trial to get accurate decodering in the learning: the implication is that this component is "memorized", and there is not a clear general trend across  signals.
(see the feature-loo-cv branch)


## older questions that might become relevant again:

- would it be worth computing an STRF: optimize response with convolved
spectrogram

- is there anything in the online analysis rate of increase that
  seems connected to differences in the behavioral data that merve saw?
    (i.e. compare online regression coefficinet magnitude as a function of condition)

  does the "test" condition vary in the focus of the source?
    - to ask this: do we see a shift across large time slices in the winning
    trained decoder (male, fem1, fem2) that is more pronounced for
    the test vs. object condition

  it's important to note that the rate of true positives is quite
  similar across the conditions: what differs is the number of
  *false* positives. This suggests that most of the differences in
  the EEG response might be found during trials without a target
  (maybe something about what a correct and incorrect trial looks like,
    maybe some resembles between truly correct and falsely correct trials)

# algorithm refinement?
- try an L1 loss
  + implement (**slow**)
  - run on cluster?
+ try L1 loss instead of cor (works very poorly)
- try L2 loss instead of cor
- cross validate regularization parameter
- try a 1-layer (should be similar to regression), and then multilayer DNN
