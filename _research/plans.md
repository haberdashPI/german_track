- how to finalize...
- double check chance performance
+ why are the timeline graphs lower in accuracy?
  + the timeline graphs appear to be accurate, I need
    to incorporate those steps into the older scripts
    to ensure they are accurate
  - fix other instances of this (under spatial, but wait until other bullet poitns are done)
- baseline: generate random windows to compare the results with
- for the best windows, using the more detailed set of bins
- plot a smooth version? (filter for visualization) or coarser grid
- early vs. late (start with no lo v hi, then do)

- split out the frequency bands, classify bin each separately
  with the above selected hyper-parameters
+ question: hit v miss? (just correct not high salience or object vs. global)

  - what to do about high 6-sec classifications?
    at the moment I'm just going to focus on earlier time points,
    but what is up with that??

# Plans

## data

- find the behavioral data
- start analyzing the subjects
- try DSS to clean the data, followed by MCCA to compress/analyze

### lower priority points
- plot results on a scalp?

### even lower priority

- check on the outlier points, maybe there is an issue with the number of data points
- try an FIR filter instead of FFT (should avoid lobes, consistent with
  paper mounya sent)

## semi-supervised decoding

as a first pass:

using nuisance mixing parameters for each windows block of the data, leave out switches, group by segments; should be linear, and so "easy" to train

first approach to solve takes up too much memory: try a solution that
uses only first-order derivative information

### more advanced semi-supervised (low priority)

employ continuous nuisance variables W of current target weightings, and apply a loss on its basis

 Σᵢ || Ab - Wᵢy ||² + γ||b||₁ + ξ||Wᵢ - Wᵢ₋₁||²

 we thus optimize over the mixing parameters of W: might need some sort of regularizer for W that has a single source bias (ala a Dirchlet distribution)

 I can work on this model by starting with an optimizer for the least squares model using an ANN, and then expanding from there.

## early vs. late targets

- analyze my current behavioral data (have done this before, just re-run)
- analyze frequency bins
- analyze decoding approach

## follow-up analyses

### baseline issues
- revisit baseline later on: baseline didn't look all that great;
  - but there is only a little but of an issue with the overall within
    trial trend (a bump near the end)
  - and a little issue across trials (some trials are quite noisy)
  - no obvious trends shown

### odds and ends
- select features (contralatera?, central/frontal?)
- error bars for individuals?
- maybe look at false alarms?


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
