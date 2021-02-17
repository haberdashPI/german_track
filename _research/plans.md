mounya thought: test decoders on condition they weren't trained on
david thought: train decoders across all conditions

### priorities

- add decoding to fig. 2
- add decoding to fig. 3
- add decoding to fig. 4

- can we see decoding when we test on different condition than training
- can we see differences when using a joint vs. split decoder across conditions?
- consider adding phase features: this could help classification and decoding
    - try it in one "easy" case, and see if it helps
        + does it help with decoding: doesn't seem to
            - however, might be useful if we actually added the azimuth
              feature for decoding
        - does it help with condition or salience classification?

## plans

- build-up angles: careful look at merve's data
- decoding: mixed and separate other targets
- decoding: timeline for target
+ conditions: condition-hit timeline
- decoding: joint mixture??
- early/late salience (but classifier trained across both cases)

remaining tasks:

1. merve's individual data for build-up angles
- still questions: data not quite the same
- intercept for logistic regression has a confusing value (why is it negative??)

codecleanup:

figure nitpicks:
- set scales (0.5 - 1 mostly)
- change y axis labels for neuarl data near/far
- use more horizontal space
- focus on angle for the buildup curve (don't need hit rate)
- include engle in fig 4 (fig 3 data)
- maybe a difference between neuarl

- feature plots for fig2 & 3?

# cleanup

- rename ishit

## 1. Main result: the different conditions have different brain responses

- [X] power analysis
- [X] subpanel broken down by different response types
- [X] establish a baseline: null model? shuffled labels? hit type?
    - [X] random pre-target window
    - [X] random post-target window
    - [X] shuffled labels: better than random chance (yes)
    - [X] null model: do features matter (yes)
    - [X] random window: does window timing matter (no)
    - [X] random trial type: does the response type matter (yes)
- [X] regression components
- [X] subset of channels
- [x] plot of main result and selected baseline !! (can occur indepndent of behavioral data)
    - [X] recent changes introduced a bug; all labels are 'object' (did we loose the others, or did they change to 'object'?)
    - [x] add baseline to main bar plot
- [x] get human data !!
    - [X] store and load raw event files
    - [x] find stimulus metadata (timings of targets; stored in .mat file)
    - [x] store metadata in portable format, and load the data
- [ ] stats
- [ ] try MCCA averaging???

