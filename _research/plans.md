presentation (for lab meeting tuesday)

what do we show:
abstract references early late
- maybe salience early/late?
- or merve's original figure for hit rate

## plans

- condition: show classifying cond-hit vs. all other, vs. cond vs. all other
    seems to work in both cases (though the stats are *bit* tricky)
    this suggests that there truly is a different configuration at the
    instant of target detection:
    we would expect to see a broadly distritubed ability to detect
    listenin condition, but a focused time window for the hit detection
    ... gah, I'm struggling to think about this right now

    - concerns:
        - i'm worried the beta regression doesn't capture this data:
            using a logistic regression allay's my concerns
          I may need to use a logistic regression
        - what does this actually proove, why did I think this was a good idea?
            - OH!! if there is a unfirom "window of attention" that moves about
              rather than one that can distribute across different expanses of
              the target, then the only thing that can differentiate
              the three listening conditions is an overall "command" signal
              a classifier of the conditions would pick up on this command signal,
              but the command signal should be essentialy the same regardless of
              whether the target is detected or not, therefore, if we can
              train a classifier that can differentiate the conditions
              *when* there is a hit, that eliminates this explanation

              furthermore, the features used by the latter classifier should
              differ from those used by the former

- build-up angles: careful look at merve's data
- early/late salience (but classifier trained across both cases)
- decoding: dnn decoding working, now get some plots going, show mounya both with and
without additional model layer

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

