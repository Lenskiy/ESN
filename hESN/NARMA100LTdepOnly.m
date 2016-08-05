% Generates NARMA100longTerm benchmark time-series
% Task: predict y(t) given u(t)
function [ O ] = NARMA100LTdepOnly( len )
   
    % init Y with 0s
    Y = zeros(len,1);
    
    % init U with uniform [0,0.5]
    U = 0.5*rand(len,1);
    
    for i = 101 : len 
        
        Y(i,1) = 0.3*Y(i-25,1)+0.005*Y(i-50,1)*sum(Y(i-100:i-50,:),1)+1.5*U(i-25,1)*U(i-100,1)+0.1;
        
    end
    
    Y = Y(101:size(Y,1),1);
    U = U(101:size(U,1),1);
   
    O = [U, Y]';
end
