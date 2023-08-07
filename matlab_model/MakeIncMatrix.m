function I=MakeIncMatrix(s,t)

    MaxNode=max(max(s),max(t));
  
    I=zeros(MaxNode,length(s));
      
    for j=1:length(s)
        I(s(j),j)=1;
        I(t(j),j)=-1;
    end

end


% julia
% function MakeIncMatrix(s, t)
%     MaxNode = max(maximum(s), maximum(t))
%     I = zeros(MaxNode, length(s))
%   
%     for j in 1:length(s)
%         I[s[j], j] = 1
%         I[t[j], j] = -1
%     end
%     
%     return I
% end