function loggedSpeech = stretch(ss,AnalysisLen,SynthesisLen)
% this uses a vocoder to stretch the duration of a stimulus based on
% the ratio of AnalysisLen and SynthesisLen (resampling can then be
% used to change the pitch)

WindowLen = 256;
Hopratio = SynthesisLen/AnalysisLen;

%%
% Create a System object to read in the input speech signal from an audio
% file.
hAudioSource = dsp.SignalSource('Signal',ss,'SamplesPerFrame',AnalysisLen);

%%
% Create a buffer System object, which is used for the ST-FFT.
hbuf = dsp.Buffer(WindowLen, WindowLen - AnalysisLen);

% Create a Window System object, which is used for the ST-FFT. This object
% applies a window to the buffered input data.
hwin = dsp.Window('Hanning', 'Sampling', 'Periodic');

%%
% Create an FFT System object, which is used for the ST-FFT.
hfft = dsp.FFT;

% Create an IFFT System object, which is used for the IST-FFT.
hifft = dsp.IFFT('ConjugateSymmetricInput', true, ...
  'Normalize', false);

%%
% Create a System object to play original speech signal.
% hAudioOut = dsp.AudioPlayer('framerate', Fs);

% Create a System object to log your data.
hslg = dsp.SignalSink;

%%
% Initialize the variables used in the processing loop.
yprevwin = zeros(WindowLen-SynthesisLen, 1);
gain = 1/(WindowLen*sum(hanning(WindowLen,'periodic').^2)/SynthesisLen);
unwrapdata = 2*pi*AnalysisLen*(0:WindowLen-1)'/WindowLen;
yangle = zeros(WindowLen, 1);
firsttime = true;

%% Stream Processing Loop
% Now that you have instantiated your System objects, you can create a
% processing loop that performs time stretching on the input signal. The
% loop is stopped when you reach the end of the input file, which is
% detected by the |AudioFileReader| System object.
while ~isDone(hAudioSource)
    y = step(hAudioSource);

%     step(hAudioOut, y);    % Play back original audio

    % ST-FFT
    % FFT of a windowed buffered signal
    yfft = step(hfft, step(hwin, step(hbuf, y)));

    % Convert complex FFT data to magnitude and phase.
    ymag       = abs(yfft);
    yprevangle = yangle;
    yangle     = angle(yfft);

    % Synthesis Phase Calculation
    % The synthesis phase is calculated by computing the phase increments
    % between successive frequency transforms, unwrapping them, and scaling
    % them by the ratio between the analysis and synthesis hop sizes.
    yunwrap = (yangle - yprevangle) - unwrapdata;
    yunwrap = yunwrap - round(yunwrap/(2*pi))*2*pi;
    yunwrap = (yunwrap + unwrapdata) * Hopratio;
    if firsttime
        ysangle = yangle;
        firsttime = false;
    else
        ysangle = ysangle + yunwrap;
    end

    % Convert magnitude and phase to complex numbers.
    ys = ymag .* complex(cos(ysangle), sin(ysangle));

    % IST-FFT
    ywin  = step(hwin, step(hifft,ys));    % Windowed IFFT

    % Overlap-add operation
    olapadd  = [ywin(1:end-SynthesisLen,:) + yprevwin; ...
                ywin(end-SynthesisLen+1:end,:)];
    yistfft  = olapadd(1:SynthesisLen,:);
    yprevwin = olapadd(SynthesisLen+1:end,:);

    % Compensate for the scaling that was introduced by the overlap-add
    % operation
    yistfft = yistfft * gain;

    step(hslg, yistfft);     % Log signal
end

%% Release
% Here you call the release method on the System objects to close any open
% files and devices.

release(hAudioSource);
loggedSpeech = hslg.Buffer(200:end)';

