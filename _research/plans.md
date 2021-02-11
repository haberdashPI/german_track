cleanup:
    - use more efficient arrow file storage format
    - store the results in the cache, not dvc, it's too transient

mounya thought: test decoders on condition they weren't trained on
david thought: train decoders across all conditions

### options for ARO

option 0:

a mix of 1 and 2: we make all the slides mounya asked for, and we provide
some indication of the flicker/spread dichotomy

-

options 1:
- intro becomes clear: frame in terms of spotlight hypothesis
    - present timeline shifting hypothesis
- just use global and object (to avoid issues with spatial analysis)
- we can also note that for both conditions, the decoding and the attentional advantage increases across the course of the trial

hold on: don't both hypothesis predict the pretarget-attend window will be smaller for global... a more nuanced version could
what we're looking for is a form of "flickering"
other measures of this could be
    1. flicker
        - variance across time would be greater for global condition
        - high correlations should cluster: i.e. log-gamma disribution vs. exponential distribution
        - overall variance would be lager for the global condition
    2. spreading would
        - variance across time would be similar across conditions
        - we should see an overall increase in correlation to the other targets
        for the global condition compared to the object and spatial conditions
        - overall variance would be less for the global condition


what it is flickering? its when the variance across the entire attentional signal
is greater than variance in a local neighborhood

a key point is that the decoding signal appears to be distributed similarly across these conditions and their timelines are remarkably similar to one another

NOTE: after thinking through this more, we still have a lot to figure out
about this, it's not ready for prime-time

THOUGHT: one possible source of evidence for spread would be if there is an indication that the athit-other is higher for the global vs. the object and spatial condition near the target - this would need to specifically be an interaction term, because there is probalby an overall effect across conditions, based on the figures

What we really want is a control condition where we are relatively confident that participants are not flickering during the object condition, then we could say that for these cases there is (or is not) evidence for flickering relative to that baseline

overall impression after looking at this for a while: the attentional signal is remarkably similar across the three conditions: given that there is reasonable variance in the signal (is there some way to quantify this relative to some reference?) this suggests that there is a "sampling" or "flickering" in all three conditions, that the task is hard enough to prevent a steady attentinoal state for any of the conditions, and further that whatever differences in behavior that we see arise mostly from selective processing that occurs following the signal we are picking up on using the decoder.

option 2:
simplest interpretation of Mounya's suggestion
- intro:
    what can we say here?
    - we can talk about why this design is interesting
      - typical decoding experiments ask participant to attend to
      one source throughout the experiment
      - in the present case we consider several different attention states
        can we read the state, and see differences here
     - it is usually easy for subjects to attend to a given source
       (so start of trial is as good as attention is likely to get)
     - here the task is intentially quite challengin, in the hopes that
        we can see listener's "honing-in" on sources of interest

main result: establishes that participant do indeed treat the conditions differently
neural result: we see evidence for distinct configurations specific to trial hits
    - training on hits, we see differences across the three conditions
    - differences between more global and more selective listening are largely due to coefficients in theta and alpha frequency ranges
    -

- present stats for main result
- present baseline models, and show the features used by the classifier
    and so what: what's the main point here - we see an indication of
    some "instructional" signal
        - we see this is selective to specific frequency bins when comparing
            distributed vs. selective listening conditions
- bonus:
    - show behavioral switch data
    - show decoding accuracy: we can reliably decode the target sources
    - show decoding accuracy over target time
        - we can see improvements in decoding accuracy over the course of the trial
        - we see increasing advantage of the target over other sources

option

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

