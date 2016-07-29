function [ E ] = RMSE( in1, in2 )
% Calculates RMSE
    
    E = sqrt(sum((in1-in2).^2)/length(in1));

end

