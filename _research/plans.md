# meeting (3/22)
- show for target vs. other sources
    - right now my results actually ignore the
      target when the "% correct" responses.
- cleanup: get both types of decoding working together
- look at misses
- look at alingment to target
- aling to the switches, look at the later switches (do they attend to the "right" target)
- plot x = time, y = hitrate across trial time
- spatial: have a different feature of pitch for each source

trials matched: hit
same trial, same acoustics
only trials that have late targets
far from a switch

decoding after the target
decoding before the target (timeline locked to the target)

can't blame different amounts of data: matched
across conditions

if there's a match near target, and mismatch earlier

but then they turned at the right moment


the things to do:

1. redo: analysis by spatial: connects
with prior literature

2. new analysis we're just describing: can we support roving vs. switching

# meeting (3/16)

- redo decoder in range 0 - 6.5 (minimum trial length?)
    (exclude training after that)
    TODO: can we plot this data different first, to just see if
    it's an issue of the reducing number of cases
    and not an issue of training
- can we move the bump in decoding by splitting trials
    between early/late for a given switch (e.g. the first switch "bump")
- decoding accuracy "out of the three possible decoders which is better"
    per trial per time instant,
    what range of accuracies do we get
    OBJECT CONDITION ONLY

do we still see these bumps, and do they relate to the switches
individual level for each switch

for the spatial condition: use the channels as the "target" source

maybe global condition: one with each source, one with each channel (left/right)
global: 1 of 3, spatial: 1 of 2

# meeting

1. how much does the lag matter for these new figures
    - look at differences
2. are any of the patterns related to switches?
    work with these curves, show distribution of switches
    or on a per trial (align to switch onset)

for behavior:
    focus on hr fr, split fr

talk to angie about what her experience
    - what does startup look like
    - support more generally

talk to nick: get his perspective (works at an EEG, smaller)
# priorities

first track:
- establish: why is it so different from older results
- then establish roving behavior

parallel, second track:
- cleanup of existing results

## differ from old results
- try aggregating results as follows
    - for each speaker model: how does the resulting stimulus compare
        to the trained vs. untrained speakers
    - do with time points aligned relative to the start of a trial
    - do with time points aligned to switches?

look at things per trial Q: why so different from traditional decoding

try looking at a object condition trained on the male speaker vs. each female speaker
compare that to the global condition

## roving behavior
once we've established whether we can get similar results, then do a "hoping" check

## cleanup of existing results

fig2: show variants on d' send to mounya
show fp broken down seperately

for fig4:
run the slopes analysis with new data organization
try hit rate by target time (if that's good, try hit rate by decoding accuracy)

alpha oscillation - look at alpha power or gamma power or global or object
    - time lock to target time

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

