
function alert(message)
    if nargin == 0
        message = 'Done!'
    end
    if ismac
        system(['osascript -e ''display notification "' message '" with title '...
                '"Matlab Language" sound name "Submarine"'''])
    else
        msgbox(message);
    end
end
