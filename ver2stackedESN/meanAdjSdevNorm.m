% Centeres with mean and normalizes with sdev the input series 
function [ Y ] = meanAdjSdevNorm( X )
    
    Y = (X - mean(X))/std(X);

end

