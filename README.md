# German Track

This project aims to decode EEG signals from a three-speaker experiment that
includes three listening conditions: listening to all speakers--the *global*
condition--listening to one speaker--the *object* condition--or listening to
one ear--the *feature* condition.

In some places throughout the code, due to unfortunate label usage in the original experiment, the `test` condition is used to indicate the `global` condition. (TODO: At some point I need to have the `test` label renmaed throughout most of the code)

TODO: more details about the project should eventually go here.

## Download Source Code and Data

The project uses [git](https://git-scm.com/) to manage versions of the source code and
[dvc](https://dvc.org/doc/start/data-versioning) to manage versions of the data. You will
need to install both to use this project. Once you have theese tools installed, setup the
project as follows:

```bash
git clone https://github.com/haberdashPI/german_track
cd german_track
```

The above commands will create the project folder and open it. Next, download the [project
data](https://osf.io/rsfm5/). Store it in a folder somewhere *outside* of the `german_track`
folder you just created.

Then use the folder in the commands below, to setup data version controll (`dvc`).

```bash
dvc remote add [local data dir]
dvc pull
```

## Installation

To setup these analysis scripts on a new computer:

1. Install [MATALB](https://www.mathworks.com)
2. Install [R](https://www.r-project.org)
3. Install [Julia](https://julialang.org)
4. Download [fieldtrip](http://www.fieldtriptoolbox.org/download/) and add it to the MATLAB path.
5. Download [NoiseTools](http://audition.ens.fr/adc/NoiseTools/src/) (Version from 18-Feb-2020) and add it to the MATLAB path.

Run `scripts/julia/install.jl` in julia.

## Project organization

- `scripts` The top-level scripts called to analyze the data.
- `src` all supporting code called from the top-level scripts
- `data` all of the raw and processed experimental data (managed using `dvc`)
- `plots` the code to generate plots, and their resulting pdfs
- `_research` contains various temporary files
- `notebooks` will contain any notebooks with plots / analyses in them
- `papers` will contain research papers on this project
- `test` any unit tests (almost nothing at this point)

### Notes on specific files in `scripts`

- `matlab/generate_stimuli` - Used to create the experimental stimuli. *Read comments carefully* if you regenerate the experimental stimuli.
- `matlab/read_eeg_events.m` - Load the event markers for the onset of each stimulus.
- `R/read_sound_events.R` - Further processing of events to combine them with
  the event files which describe which stimuli were presented when.
- `matlab/read_eeg_response.m` - Load and preprocess the data for the EEG channels
- `matlab/clean_data.m` - manually clean egregious artifacts from data.
- `matlab/clean_with_mcca.m` - following manual cleaning, clean using MCCA analysis.
- `R/test_[other].R` - all files with this prefix are out of date, and should not be used
- `julia/test_condition.jl` - run a static analysis over each condition
- `julia/test_online.jl` - run several online anlayses over each condition
- `julia/test_[other].jl` - these files are out of date

### Notes on specific files in `src`

**TODO**

## Regenerating processed data

### EEG data

There are few steps necessary to regenerate the preprocessed data files (which
are stored in the `data/processed` subfolder).

1. Call `scripts/matlab/read_eeg_events.m` to generate *.csv files with the event triggers.
2. **Optional** comment out the call to `redatedir` to generate the data in the same output directory as used previously.
3. Call `scripts/R/read_sound_events.R` to filter the events based on the
   Presentation log file. The result will be a set of 150 events, corresponding
   to the start of the 50 trials for each of the three conditions. This script
   should probably be run incrementally if you add any subjects: i.e. copy the
   body of the for loop, setting sid manually. On each run
   verify the output, as you run it. (e.g. there is a graph that gets generated
   of all events in the EEG file). It should show three clear breaks
   in timing corresponding to the three conditions run. There should be a
   total of 50 relevant events after each of the breaks (before the first break,
   the training trials are presented, and those are ignored).
4. Call `scripts/matlab/read_eeg_response.m` to generate the `*.mat` files
   with the preprocessed event streams.
5. Call `scripts/matlab/clean_data.m` and run through the steps manually to eliminated any egregious artificats.
6. Call `scripts/matlab/clean_with_mcca.m` and run through the steps manually to generate data cleaned with MCCA.

### Pitch Estimates

I used [CREPE](https://github.com/marl/crepe) to estimate the pitch of each speaker. Follow the directions there to install CREPE and then run the file
`scripts/julia/find_pitches.jl`

## Subject notes

The first three subjects' events are likely poorly aligned due to some
issues with the sound drivers on the computer presenting stimuli.

Subjects 1-7 used an old stimulus set, for which we do not have ground
truth of the target modulation or HRTF transformed sentences. These subjects
data must also be loaded in a slightly different way, which can be found by
checking out older versions of the repository.

Subjects 8-10: ex1,3,5 swapped (on right instead of left side)
Subjects 8-16: A28 is a bad electrode.

Subject 9 used an older version of the trigger codes, and the first test
trial (of 150) was lost (the scripts loading 9's data account for this).

Subject 24: Recorded start of EEG data using wrong config: this is stored as
24A (under the `ignore` folder); this recording occurred during the training
section of the experiment; since training data is already ignored, no change in
the analysis for this subject is necessary.

