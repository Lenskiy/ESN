% Generates Fibonacci time-series
function [ O ] = Fib( len )
   
    % init Y with 0s
    Y = zeros(len,1);
    U = zeros(len,1);
    
    Y(1,1) = 1;
    Y(2,1) = 1;
    U(1,1) = 1;
    U(2,1) = 2;
    
    for i = 3 : len 
        
        Y(i,1) = Y(i-1,1)+Y(i-2,1);
        U(i,1) = i;
    end
    
   
    O = [U, Y]';
end
