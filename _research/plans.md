
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

- do the features encoded tell us anything
  - look at bootstraps
    - across time
    - across features (ideally spatialized)
  
- what about comparing across conditions: e.g. train on global test on 
  same stim-id for object (that should tell us if there is something different)

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
