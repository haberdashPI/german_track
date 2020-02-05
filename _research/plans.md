
# Plans

## current steps:

+ hits vs misses (focus on that, as defined by task)
+ baseline - find something away from target or switch
   - misses should match baseline
    - doesn't seem to be the case, exactly; looks pretty noisy
      worth investigating individual data points, at the moment
      I would feel pretty unconvinced that there's any effect
- pair data - lines between the two conditions (or some such)
- check on the outlier points, maybe there is an issue with the number of data points
- revisit window timings (start & length) / frequency bands

later on:
- early vs. late?
- high v low salience?
- select features (contralatera?, central/frontal?)
- error bars for individuals?
- maybe look at false alarms?

analysis:

if we get something with the above analysis it might be worht consider a semi-supervised approach

- semi-supervised training, with this approach mounya suggested
  that the loss could be the maximum loss of the three speakers
  at each time point (could also have some sort of continuity cost:
  i.e. transitions in attention are assumed to be more gradual)

employ a nuisance variable W: the current target weightings, and apply a loss
on its basis

 Σᵢ || Ab - Wᵢy ||² + γ||b||₁ + ξ||Wᵢ - Wᵢ₋₁||²

 we thus optimize over the mixing parameters of W: might need some sort of regularizer for W that has a single source bias (ala a Dirchlet distribution)

 I can work on this model by starting with an optimizer for the least squares model using an ANN, and then expanding from there.


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
