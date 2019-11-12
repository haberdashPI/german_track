# German Track

This project aims to decode EEG signals from a three-speaker experiment that
includes three listening conditions: listening to all speakers--the *global*
condition--listening to one speaker--the *object* condition--or listening to
one ear--the *feature* condition.

In some places throughout the code, due to unfortunate label usage in the original experiment, the `test` condition is used to indicate the `global` condition. (TODO: At some point I need to have the `test` label renmaed throughout most of the code)

TODO: more details about the project should eventually go here.

## Installation

To setup these analysis scripts on a new computer:

1. Install [MATALB](https://www.mathworks.com)
2. Install [R](https://www.r-project.org)
3. Install [Julia](https://julialang.org)
4. Install [fieldtrip](http://www.fieldtriptoolbox.org/download/) in Matlab
5. Create a file called `install.toml` in the base directory containing `data = "[data dir]"` with `[data dir]` replaced with the directory containing the preprocessed data (stored separately from the git repository)

Run `scripts/julia/install.jl` in julia.

## Project organization

- `scripts` The top-level scripts called to analyze the data.
- `src` all supporting code called from the top-level scripts
- `data` all of the raw and processed experimental data
- `plots` the code to generate plots, and their resulting pdfs (mostly in R)
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

There are few steps necessary to regenerate the preprocessed data files
(which are stored in the `data/exp_pro` subfolder). In
`data/exp_raw/config.json` you can specify the location of the raw BDF files
and Presentation *.log files for your local machine. Once specified, you can
use the following steps to generate the preprocessed data on your computer.
This pipeline will skip file generation if it finds existing files in the
output directory so you can also use this pipeline to add preprocessed data
for a new participant, by including their raw BDF file in the same location
as all other participant's raw data.

1. Call `scripts/matlab/read_eeg_events.m` to generate *.csv files with the event triggers.
2. **Optional** comment out the call to `redatedir` to generate the data in the same output directory as used previously.
2. If you skipped step 2, update the dated directory in `dateconfig.json`.
   It should be equal to the directory specified by `data_dir` after running `read_eeg_events.m`.
3. Call `scripts/R/read_sound_events.R` to filter the events based on the
   Presentation log file. The result will be a set of 150 events, corresponding
   to the start of the 50 trials for each of the three conditions. This
   script must be run incrementally: i.e. copy each section of code to R
   and verify the output, as you run it. (e.g. there is a graph that gets
   generated of all events in the EEG file).
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

