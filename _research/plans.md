
# Plans

## checks
+ verify `test_correct`

+ analyze behavioral data

## new analysis config
- do we compare across conditions (with same stimulus and subject)
    - compare decoders across conditions (same stimulus)
    - does the target event get detected better across conditions?
    - distinguish decoding between male and female targets across object and test conditions
      - try out the additional encodings (alpha & theta)
    - in online decoding, does the increase look different?
- do we compare across times (e.g. decoding better near target)
  + does decoding differ by target locus
  - does decoding differ near switches


  questions:

  what actual features are being used in the decoders?
    - across 

  measure: advantage of decoding at target vs. non-target
  is this advantage greater for male vs. female in the object condition
  is this advantage similar for male vs. female in the test condition

  measure: decoding of target
  is decoding of the male target target better for the object vs.
  test condition?

  do we see a shift across large time slices in the winning
  trained decoder (male, fem1, fem2) that is more pronounced for
  the test vs. object condition

  wait... how does the correct response work into this?
  look at the behavior data, and rethink this issue

  it's important to note that the rate of true positives is quite
  similar across the conditions: what differs is the number of 
  *false* positives. This suggests that most of the differences in
  the EEG response might be found during trials without a target
  (maybe something about what a correct and incorrect trial looks like,
    maybe some resembles between truly correct and falsely correct trials)

  does accuracy differ for male and female targets?

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
