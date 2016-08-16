function [x,y]=hilbert(n)
%HILBERT Hilbert curve.
%
% [x,y]=hilbert(n) gives the vector coordinates of points
%   in n-th order Hilbert curve of area 1.
%
% Example: plot of 5-th order curve
%
% [x,y]=hilbert(5);line(x,y)
    step = 16;
    if n<=0
        x=0;
        y=0;
    else
        [xo,yo]=hilbert(n-1);
        x=0.5*[-step+yo -step+xo step+xo  step-yo];
        y=0.5*[-step+xo  step+yo step+yo -step-xo];
    end
end