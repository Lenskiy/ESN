function value=getoption(options, field, defaultvalue)
% function value=getoption(options, field, defaultvalue);
% Get a value from a structure options with defaultvalue
%

if nargin<3
    defaultvalue=[];
end
if isfield(options,field)
    value=options.(field);
else
    value=defaultvalue;
end