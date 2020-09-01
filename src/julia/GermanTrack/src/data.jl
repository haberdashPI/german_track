
# Correctly interprets a given row of the data as a hit, correct rejection, false positive
# or miss. Since the directions are different for each condition, how we interpret a an
# indication of a detected target depends on the condition.

function ishit(row; kwds...)
    vals = merge(row,kwds)
    if vals.target_present
        if vals.condition == "global"
            vals.target_detected ? "hit" : "miss"
        elseif vals.condition == "object"
            vals.target_source == "male" ?
                (vals.target_detected ? "hit" : "miss") :
                (vals.target_detected ? "falsep" : "reject")
        else
            @assert vals.condition == "spatial"
            vals.direction == "right" ?
                (vals.target_detected ? "hit" : "miss") :
                (vals.target_detected ? "falsep" : "reject")
        end
    else
        vals.target_detected ? "reject" : "falsep"
    end
end
