### priorities

we want to see if there are any differences in the conditions' timeline
- look at coefficients averages by lag
    - definitely examine by hit and miss (maybe hits are faster for object because that's the only case where it works)
    - compare across conditions
- look at timeline of decoding accuracy
    - are there more fluctuations for the global decoding (would suggest more movement)
        - do we see this in individual trials
        - do we see an increase in the power of particular frequencies of the correlation
        over time
    - question: do we gain something by decoding pitch and envelope together
        - not with 64 hidden units: try 128 units? no, doesn't work up to 1024 units
- can we see differences when using a joint vs. split decoder across conditions?
- consider adding phase features: this could help classification and decoding
    - try it in one "easy" case, and see if it helps
        - doesn't help with decoding
        - does it help with condition or salience classification?

###

- decoding: look atmiss-miss
- look at target classification vs. pre-target window region, compare cacross codnitions,
use a baseline with two random windows before the target

presentation (for lab meeting tuesday)

what do we show:
abstract references early late
- maybe salience early/late?
- or merve's original figure for hit rate

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

