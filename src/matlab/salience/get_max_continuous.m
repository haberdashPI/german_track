function [prev_start, prev_max] =get_max_continuous(input_array)
    prev_start=1;
    prev_max=0;
    start=1;
    max_l=0;
    started=1;
    for k=1:length(input_array)
        if (input_array(k) && started)
            max_l=max_l+1;
        elseif (input_array(k) && ~started)
            started=1;
            max_l=1;
            start=k;
        else
            started=0;
        end
        if max_l>prev_max
                prev_max=max_l;
                prev_start=start;
        end
        
    end
    
end