
# Plans

## current steps:

- consider using envelope surprisal
- for plots: pick only the comparisons that we know make sense to look at 
- look at alpha and gamma bands for the spatial task
- train across subjects (exclude 11?)
- train across conditions: weight the data based on the condition

- look at patterns of the features (pick what we want to compare here as well)
  - plot by scalp position

- can we see changes in what features mater:
  hypothesis: if the dominant voice is more highly represented,
  we should see a greatly increased representation of the given voice
  in longer latency features

- is it possible to train something across the conditions? (if not that makes a strong case that the features really are different)
- can we train across subjects?

## lower priority current steps

- is there anything in the online analysis rate of increase that 
  seems connected to differences in the behavioral data that merve saw?
    (i.e. compare online regression coefficinet magnitude as a function of condition)
- 

## prior analysis config

- older thinking... (might be wroth revisiting at some point)

  why is female higher:
  - okay, (on sept 29th): these graphs convince me that the major culprit is not the source to decode, but rather, the window timing. The time when
  the female voice is a target leads to better correlations. THat suggests
  participants are more attentive to that window of time. (would be easy to test
  for by inverting the conditions of the experiment: i.e. make the female speaker the target; this should change the results)

  also here's another point to contend with: there are fewer detected female speakers in the object condition: however the ones that do get detected are quite highly correlated: this makes sense; when the female speaker is well decoded just before the target event, a listener is more likely to notice this change, even if they are not committing the same resources to that speaker

  NOTE: to be absolutely double-double sure, check out some stats of the features; if anything these would suggest that the male voice
  should be easier to encode (since there is more variance)
  - that seems to indicate some encoding differences, but if anything these would imply that the male speaker should be easier to detect (it has more feature variance)
  - this could also explain the lower response in the male target: 
    a lower SNR can be detected
 
- what about comparing across conditions: e.g. train on global test on 
  same stim-id for object (that should tell us if there is something different)
  - in more detail: test generalization of decoder trained on global correct,
    male targets (and female targets), does it work for the feature condition
    and vice versa
  - this seems to indicate the features used across conditions cannot
    be generalized: next step should be to look at those features in detail
  
- what about mounya's point about the female "correct" responses counting
  as incorrect

- do the features encoded tell us anything?
  - look at averages across time / feature
  - look at bootstraps
    - across time
    - across features (ideally specialized, but first without specialization)
 
### even older

- do we compare across conditions (with same stimulus and subject)
    - compare decoders across conditions (same stimulus)
    - does the target event get detected better across conditions?
    - distinguish decoding between male and female targets across object and test conditions
      - try out the additional encodings (alpha & theta)
    - in online decoding, does the increase look different?
- do we compare across times (e.g. decoding better near target)
  + does decoding differ by target locus
  - does decoding differ near switches

  Note: would be worth doing a 10-fold validation to verify the meaningfulness of correct v incorrect (leave one out might be too forgiving)

  questions:

  what actual features are being used in the decoders?

  measure: advantage of decoding at target vs. non-target
  is this advantage greater for male vs. female in the object condition
  is this advantage similar for male vs. female in the test condition

  measure: decoding of target
  is decoding of the male target target better for the object vs.
  test condition?

  does the "test" condition vary in the focus of the source?
    - to ask this: do we see a shift across large time slices in the winning
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

  - does accuracy differ for male and female targets?

# algorithm refinement
- try an L1 loss
  + implement
  - run on cluster
+ try L1 loss instead of cor
- cross validate regularization parameter
- try a 1-layer (should be similar to regression), and then multilayer DNN
