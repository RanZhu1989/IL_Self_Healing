function [Mask] = MakeMask(x,y,list)
    Mask = zeros(x,y); 
    for i=1:x
        Mask(i,find(list==i))=1;
    end
end

