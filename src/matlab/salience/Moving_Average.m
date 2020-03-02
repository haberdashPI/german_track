function [Output] = Moving_Average(Input,Window)
Flip_Flag = 0;
if size(Input,2) == 1
    Input = Input';
    Flip_Flag = 1;
end


if Window == 1
    Output = Input;
    return
end
Output = zeros(size(Input));
E = size(Output,2);

for iTime = 1:(Window/2)
    Output(:,iTime) = mean(Input(:,1:(iTime + Window/2)),2);
    Output(:,E - iTime+1) = mean(Input(:,(E - iTime+1 - Window/2):E),2);
end
%Actually, window is window + 1...
for iTime = (Window/2 + 1):(E - Window / 2)
    Output(:,iTime) = mean(Input(:,(iTime - Window/2):(iTime + Window/2)),2);
end

if Flip_Flag
    Output = Output';
end