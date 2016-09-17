function [ E ] = NRMSE( in1, in2 )
% Calculates NRMSE
% in1: prediction, in2: target
    
    E = sqrt(sum((in1-in2).^2)/length(in1))/std(in2);

end

