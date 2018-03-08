CreateDemoSignals
trfTrainingWindow = attentionDuration / 20;
iTrain = find(recordingT > attentionDuration - trfTrainingWindow & ...
              recordingT < attentionDuration + trfTrainingWindow);
iTest = find(recordingT > 2*attentionDuration);

% Now calculate the models for the attended and unattended signals.
trfDirection = -1;
Lags = 0:round(1.5*impulseLength*fs);
Method = 'Shrinkage';
K = [];
doScale = 1;
attentionModel = FindTRF(attendedAudio(iTrain), response(iTrain, :), ...
    trfDirection, [], [], Lags, Method, K, doScale);
unattentionModel = FindTRF(unattendedAudio(iTrain), response(iTrain, :), ...
    trfDirection, [], [], Lags, Method);

[~, attendedPrediction] = FindTRF([], [], ...
    trfDirection, response, attentionModel, Lags, Method, K, doScale);
[~, unattendedPrediction] = FindTRF([], [], ...
    trfDirection, response, unattentionModel, Lags, Method, K, doScale);

ca = corrcoef([attendedAudio(iTest) attendedPrediction(iTest)]);
cu = corrcoef([unattendedAudio(iTest) unattendedPrediction(iTest)]);
fprintf('Attended correlation: %g, Unattended correlation: %g.\n', ...
    ca(1,2), cu(1,2));

%%
% Plot the predicted stimuli
clf
attentionSwitchPick = 3;
iPlot = find(recordingT>attentionSwitchPick*attentionDuration-trfTrainingWindow & ...
    recordingT < attentionSwitchPick*attentionDuration+trfTrainingWindow);
plot(recordingT(iPlot), [attendedPrediction(iPlot) unattendedPrediction(iPlot)]')
legend('Attended Signal', 'Unattended Signal');
axis tight
