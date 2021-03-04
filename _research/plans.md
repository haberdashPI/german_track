### priorities

- establish: why is it so different

look at things per trial Q: why so different from traditional decoding

- can we see better before target decoding accuracy
- during switch (i.e. are switches messing up tracking, so we'd see better tracking pre-target for regions far from switches, or near late switches)

simple approaches:
plot decoding accuracy timeline near/far from switch
plot decoding accuracy timeline for early/late target trials
plot accuracy at the interaction of these two

or reverse engineer:
- what condition give before target decoding accuracy that's better
take all decoding

try looking at a object condition trianed on the male speaker vs. each female speaker

2:

then some hoping check

for fig2:
show d' figure
show fp broken down as a separate figure

for fig4:
maybe show hit rate by target time
maybe show some switches along the course of the target time

alpha oscillation - look at alpha power or gamma power or global or object
    - time lock to target time

+ cleanup: use dvc to manage resampled eeg with freqbins
- cleanup: do we really need the L1 regularization??? (I *think* so, but verify)

- todo: re-run cross-validated λ selection with a better range

+ can we see decoding when we test on different condition than training: yes
- do we see accurate decoding outside of the target window?
    - try mounya's idea: what percentage of trials have an effect pre-target, does that differ by condition?
    - my idea: how often does target (and other sources) fall outside CI of random stimulus decoding pre-target? does this differ by condition?

- do the atmiss-target conditions look different for object and spatial
    merely because they are a different set of target
    i.e. would we see decodign similar to global if we treated the false
    targets as targets

- can we see differences when using a joint vs. split decoder across conditions?
    doesn't matter, we saw generalization

- consider adding phase features: this could help classification and decoding
    - try it in one "easy" case, and see if it helps
        + does it help with decoding: doesn't seem to
            - however, might be useful if we actually added the azimuth
              feature for decoding
        - does it help with condition or salience classification?

## plans

- build-up angles: careful look at merve's data
+ conditions: condition-hit timeline
- early/late salience (but classifier trained across both cases)
    - these results don't make much sense to me

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

