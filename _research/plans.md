
# Plans

## current steps:

- look at a decoder trained across conditions but within each voice
- double-check all labels of the spatial condition
+ look at alpha and gamma bands for the spatial task (didn't seem to help)
+ compare distinct timings (early and late)
+ train across conditions: weight the data based on the condition
  + how does this work without the weights (about the same)

- train across subjects (exclude 11?)
- look at features again
- look at feature diffs
  - for within condition training (1 cond)
  - for cross condition training (3 cond)
  - for trained on 2 conditions
- test generalization of 2 condition decoder to untrained condition

- test
- plot features on scalp

- look at patterns of the features (pick what we want to compare here as well)
  - plot by scalp position

- can we see changes in what features matter:
  hypothesis: if the dominant voice is more highly represented,
  we should see a greatly increased representation of the given voice
  in longer latency features

- is it possible to train something across the conditions? (if not that makes a strong case that the features really are different)
- can we train across subjects?


- NOTE: envelope surprisal doesn't seem to change things much
  (maybe not *too* surprising given that we have estimate of uncertainty)

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

### even older

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

# algorithm refinement
- try an L1 loss
  + implement (**slow**)
  - run on cluster?
+ try L1 loss instead of cor (works very poorly)
- try L2 loss instead of cor
- cross validate regularization parameter
- try a 1-layer (should be similar to regression), and then multilayer DNN
