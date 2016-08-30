% Generates NARMA10 benchmark time-series
% First 10 y are initalized as 0 (However, not specified in paper)
function [ Y ] = NARMA10series( len )
   
    % init Y with 0s
    Y = zeros(len,1);
    
    % init U with uniform [0,0.5]
    U = 0.5*rand(len,1);
    
    for i = 11 : len 
        
        Y(i,1) = 0.3*Y(i-1,1)+0.05*Y(i-1,1)*sum(Y(i-10:i-1,:),1)+1.5*U(i-1,1)*U(i-10,1)+0.1;
        
    end
    
    Y = Y(11:size(Y,1),1);

end
