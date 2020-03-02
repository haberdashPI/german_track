function [Resp_Cat] = Load_Data_Binary
    
%%%%%%%

f1 = fopen('Response_Data.bin','r');
f2 = fopen('Response_Info.bin','r');


[temp] = fread(f2,2,'uint16'); %first two numbers are fs and ntrials

fs = temp(1);
nTrial = temp(2);
clear temp
Resp_Cat = cell(1,20);
for iTrial = 1:nTrial
    temp = fread(f2,5,'uint16'); %[iSub, iTrial, Left, Right, length(Response_Binned)]
    iSub = temp(1);
    Left = temp(3);
    Right = temp(4);
    Length = temp(5);
    Resp_Hold = fread(f1,Length,'int8');
    
    Resp = Resp_Hold';
    Cat_Length = size(Resp_Cat{Left},2);
    if Cat_Length > Length
        Resp = [Resp nan(1,Cat_Length - Length)];
    elseif Length > Cat_Length && size(Resp_Cat{Left},1) > 0
        Resp_Cat{Left} = [Resp_Cat{Left} nan(size(Resp_Cat{Left},1),Length - Cat_Length)];
    end
    Resp_Cat{Left} = [Resp_Cat{Left}; -Resp];
    
    Resp = Resp_Hold';
    Cat_Length = size(Resp_Cat{Right},2);
    if Cat_Length > Length
        Resp = [Resp nan(1,Cat_Length - Length)];
    elseif Length > Cat_Length && size(Resp_Cat{Right},1) > 0
        Resp_Cat{Right} = [Resp_Cat{Right} nan(size(Resp_Cat{Right},1),Length - Cat_Length)];
    end
    Resp_Cat{Right} = [Resp_Cat{Right}; Resp];
end
;