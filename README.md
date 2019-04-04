# German Track

This project aims to decode EEG signals from a three-speaker experiment
to determine which listening condition an individual is in: listening to
all speakers--the *global* condition--listening to one speaker--the *object* 
condition--or listening to one ear--the *feature* condition.

TODO: more details about the project should eventually go here.

## Stimulus Generation

To reproduce the stimuli, re-run `create_mixtures` (under `stimuli/src`)
as shown in `generate_stimuli` (also under `stimuli/src`). This will use
`config.json` to create the experimental stimuli. 

You can generate a new stimulus set, with the same configuration parameters,
but a different random seed, by using the `configure_mixtures`, (udner
`stimuli/src`) as shown in `generate_stimuli`.

## Installation

To setup these analysis scripts on a new computer, run `install.sh` (Mac OS X
or Unix) or `install.cmd` (Windows). The install script requires an
install.json file containing `{"data": "[data dir]"}` with `[data dir]`
replaced with the directory containing the preprocessed data (stored
separately from the git repository). Create this file before running the
install script. For this installation to work R must be installed, it must be
on your `PATH`, and you must have the package `rjson` installed (i.e.
`install.packages('rjson')`).

## Regenerating the preprocessed data

There are few steps necessary to regenerate the preprocessed data files
(which are stored in the `data/` subfolder). In `config.json` you can
specify the location of the raw BDF files and Presentation *.log files for
your local machine. Once specified, you can use the following steps to
generate the preprocessed data on your. This pipeline will skip file
generation if it finds existing files in `data/` so you can also use this
pipeline to add preprocessed data for a new participant, by including their
raw BDF file in the same location as all other participant's raw data.

1. Call `analyses/read_eeg_events.m` to generate *.csv files with the event triggers
2. Call `analyses/read_sound_events.R` to filter the events based on the
   Presentation log file. The result will be a set of 150 events, corresponding
   to the start of the 50 trials for each of the three conditions. This
   script must be run incrementally: i.e. copy each section of code to R
   and verify the output, as you run it. (e.g. there is a graph that gets
   generated of all events in the EEG file).
3. Call `analyses/read_eeg_response.m` to generate the `*.mat` files
   with the preprocessed event streams.

Note that some subjects require a slighlty different procedure to analyze: see
the notes below.

## Project organization

- `analyses` The top-level scripts called from Matlab or R to analyze the data.
   This includes the EEG files, event files, and a *.mat file describing the
   stimuli used in the experiment.
- `util` all supporting routines called from the top-level scripts
- `data` all of the raw experimental data
- `plots` the code to generate plots, and their resulting pdfs (mostly in R)

## Notes on specific files in `analyses`

- `read_eeg_events.m` - Load the event markers for the onset of each stimulus.
- `read_sound_events.R` - Further processing of events to combine them with
  the event files which describe which stimuli were presented when.
- `read_eeg_response.m` - Load and preprocess the data for the EEG channels
- `ica00X.m` - A series of files to remove ICA components that look like eyeblinks
   these are currently out of date

*TODO*: replace these files with new analysis pipeline (files below are out of date)

- `k_fold_test.m`, `cv_test.m`,  These are the scripts that actually
  run and test models on the individual listeners. They produce files
  that are stored in `models` and a set of intermediate analysis results
  in `cache`. The `plots` script use this `cache` to generate figures
  of the resulting analyses.

## Notes on specific files in `util`

- `train_model.m` trains a specific type of model on the eeg data, see
  documentation in file
- `test_model.m` tests a trained model on a subset of eeg data
- `prepare_data.m` select the specific set of data to use for training
- `trial_audio.m`

## Subject notes

The first three subjects' events are likely poorly aligned due to some
issues with the sound drivers on the computer presenting stimuli.

Subjects up to 7 used an old stimulus set, for which we do not have ground
truth of the target modulation or HRTF transformed sentences. These subjects
data must also be loaded in a slightly different way, which can be found by
checking out older versions of the repository.

Subjects 8-10: A28 seems to be a bad electrode

Subject 9 used an older version of the trigger codes, and the first test
trial (of 150) was lost. `read_sound_events.R` must be slighlty modified
Comment out the checks for 150 trials and align to the second rather than the
first stimulus event form Presentation: there are commented out lines of code
to do this in the file.

## Usage notes

Many the scripts cannot be simply run. Parts of the code must be run step
by step and the results evaluated manually to ensure that the outputs make
sense. Very little of the data checking is automated at this point (and probably
shouldn't be???).

