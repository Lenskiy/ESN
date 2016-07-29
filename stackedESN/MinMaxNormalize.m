% Normalize data to [0,+scaleFactor]
function [ Y ] = MinMaxNormalize( X , scaleFactor)
    mi = min(X);
    ma = max(X);
    
    Y = scaleFactor*(X-mi)./(ma-mi);

end

