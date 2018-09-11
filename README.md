# German Track
This project aims to decode EEG signals from a three-speaker experiment
to determine which listening condition an individual is in: listening to
all speakers--the *global* condition--listening to one speaker--the *object* 
condition--or listening to one ear--the *feature* condition.

TODO: more details about the project should eventually go here.

# Project organization

- `analyses` The top-level scripts called from Matlab or R to analyze the data.
   This includes the EEG files, event files, and a *.mat file describing the
   stimuli used in the experiment.
- `util` all supporting routines called from the top-level scripts
- `data` all of the raw experimental data
- `plots` the code to generate plots, and their resulting pdfs (mostly in R)

# Notes on specific files in `analyses`

- `read_eeg_events.m` - Load the event markers for the onset of each stimulus.
- `read_sound_events.R` - Further processing of events to combine them with
  the event files which describe which stimuli were presented when.
- `read_eeg_response.m` - Load and preprocess the data for the EEG channels
- `ica00X.m` - A series of files to remove ICA components that look like eyeblinks
- `k_fold_test.m`, `cv_test.m`,  These are the scripts that actually
  run and test models on the individual listeners. They produce files
  that are stored in `models` and a set of intermediate analysis results
  in `cache`. The `plots` script use this `cache` to generate figures
  of the resulting analyses.

# Notes on specific files in `util`

- `train_model.m` trains a specific type of model on the eeg data, see
  documentation in file 
- `test_model.m` tests a trained model on a subset of eeg data
- `prepare_data.m` select the specific set of data to use for training
- `trial_audio.m` 

# Subject notes:

The first three subjects' events are likely poorly aligned due to some
issues with the sound drivers on the computer presenting stimuli. 

# Usage notes

Many the scripts cannot be simply run. Parts of the code must be run step
by step and the results evaluated manually to ensure that the outputs make
sense. Very little of the data checking is automated at this point (and probably
shouldn't be???).

