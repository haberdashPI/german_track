
# Plans


## checks
+ verify `test_correct`

- analyze behavioral data

## new features
- filter eeg data using 
  - alpha for spatial?
        - in Frey et al 2014 (J Neuro), they find the band to be around 8-16 Hz (looked at synchronization (e.g. phase coherence))
        - in Müller and Weisz 2012 (C Cortex) in the range of 5-15, and further filtered to 9 ± 3 (looked at power)
        - in region of 11 ± 3 with filtering (from a window of 6 - 15 Hz)
    - approach: use fft on trial? (sftf?)
        correlate alpha power with envelope?
        correlate filtered signal with envelope? (seems unlikely)

  - gamma for others?
        - 30-50 Hz (need to use higher sampling rate)
        - (we need more specific measures here, can't really get an envelope following response here)
    
    - approach: filter the signal within this range
        correlate alpha power, phase coherence within this range

- analyze sound using:
    - mean while, include multiple frequency bands of stimulus (check out ed lalor's recent publications)
    - and a pitch analysis 
    - stimulus encoding: consider multiple frequency bands
    - include surprisal (e.g. pitch derivative) for target

## new analysis config
- do we compare across conditions (with same stimulus and subject)
    - in online decoding, does the increase look different?
    - does the target event get detected better across conditions?
- static windows around locations of interest

