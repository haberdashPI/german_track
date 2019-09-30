
# Plans

## new analysis config

- why is fem1 higher?
  - is it true of fem2?
    something about male, or something about fem1 specifically
    - in looking at this I see that:
      1. the difference in fem and male is due to two things:
        some advantage of before_fem training over before_male training
        some advantage of decoded source fem over decoded source male
      2. the advantage for decoded source fem2, if it is exists is
        between that of fem1 and male for the before_male training
        and above either fem1 or male 2 for the before_female training
      3. all of these differences aren't significant on their own
         but appear to accumulate, for the case where 1's conditions above
         are all in the right direction to show the largest differnece
  - is this something about differences in the features of the envelopes
    compute various statistics of those

  - okay, (on sept 29th): these graphs convince me that the major culprit is not the source to decode, but rather, the window timing. The time when
  the female voice is a target leads to better correlations. THat suggests
  participants are more attentive to that window of time. (would be easy to test
  for by inverting the conditions of the experiment: i.e. make the female speaker the target; this should change the results)

  also here's another point to contend with: there are fewer detected female speakers in the object condition: however the ones that do get detected are quite highly correlated: this makes sense; when the female speaker is well decoded just before the target event, a listener is more likely to notice this change, even if they are not committing the same resources to that speaker

  NOTE: to be absolutely double-double sure, check out some stats of the features; if anything these would suggest that the male voice
  should be easier to encode (since there is more variance)
 
- what about comparing across conditions: e.g. train on global test on 
  same stim-id for object (that should tell us if there is something different)
  - in more detail: test generalization of decoder trained on global correct,
    male targets (and female targets), does it work for the feature condition
    and vice versa
  - this seems to indicate the features used across conditions cannot
    be generalized: next step should be to look at those features in detail

- do the features encoded tell us anything
  - look at bootstraps
    - across time
    - across features (ideally spatialized, but first without spatialization)
 
### old

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
