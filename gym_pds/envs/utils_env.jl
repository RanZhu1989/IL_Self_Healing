# utils for building environment

# pdn data
function MakeMask(x, y, list)
    Mask = zeros(x, y)
    for i = 1:x
        Mask[i, findall(x -> x == i, list)] .= 1
    end
    return Mask
end

function MakeIncMatrix(s, t)
    MaxNode = max(maximum(s), maximum(t))
    Inc = zeros(MaxNode, length(s))
    for j = 1:length(s)
        Inc[s[j], j] = 1
        Inc[t[j], j] = -1
    end
    return Inc
end