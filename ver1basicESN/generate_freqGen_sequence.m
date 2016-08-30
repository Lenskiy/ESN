function [inputSequence, outputSequence] = ...
    generate_freqGen_sequence(sequenceLength, outMinPeriod, outMaxPeriod, superPeriod)
%  Generates an input-output sequence of the tunable frequency generator task. 

% inputs: 
% sequenceLength: a natural number, indicating the length of the
% sequence to be generated
% outMinPeriod: a natural number indicating the period length of the
%               highest frequency to be included in the task
% outMaxPeriod: a natural number indicating the period length of the
%               lowest frequency to be included in the task
% superPeriod: a natural number indicating the input period length
%
% outputs: 
% InputSequence: array of size sequenceLength x 2. First column contains bias 
%                input (all 1's), second input contains the slow sine input (normalized to 
%                range in [0,1]
% OutputSequence: array of size sequenceLength x 1 with the fast sine
%                 output


% Generated H. Jaeger June 23, 2007

% %%%% create slow sine input 
% outPeriodSetting = (sin(  2 * pi * (1:sequenceLength)' / superPeriod) + 1)/2;

%%%% create sequence of random constants as input
outPeriodSetting = zeros(sequenceLength,1);
currentValue = rand;
for i = 1:sequenceLength
    if rand < 0.015
        currentValue = rand;
    end
    outPeriodSetting(i,1) = currentValue;
end



inputSequence = 1 * [ones(sequenceLength,1)  -outPeriodSetting + 1 ];

currentSinArg = 0;
outputSequence = zeros(sequenceLength,1);
for i = 2:sequenceLength
    currentOutPeriodLength = outPeriodSetting(i-1,1) * (outMaxPeriod - outMinPeriod) + outMinPeriod;
    currentSinArg = currentSinArg + 2 * pi / currentOutPeriodLength;
    outputSequence(i,1) = 1*(sin(currentSinArg) + 1)/2;
end

