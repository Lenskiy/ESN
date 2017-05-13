%% Lorenz training/testing data
addpath('../T0_chaotic/');
addpath('../utilities/');
[x1, x2, x3] = lorenz(28, 10, 8/3);
dataLen = 20000;
startInd = 1;
Lorenz_data = [x1(startInd:startInd + dataLen - 1),...
               x2(startInd:startInd + dataLen - 1),...
               x3(startInd:startInd + dataLen - 1)]';
Lorenz_data = Lorenz_data / (std(sum(Lorenz_data)));
           

p = 0.8;
mid_point = round((length(Lorenz_data) - 1) * p);
train_input = Lorenz_data(:, 1:mid_point);
train_output = Lorenz_data(:, 2:mid_point + 1);
test_input = Lorenz_data(:, mid_point + 1:end - 1);
test_output = Lorenz_data(:, mid_point + 2:end, :);

Nsize = [4     15       36      65     102     147       200      261       330];
N_trials = 10;
radii = [0.25, 0.5, 0.75, 1.0, 1.25];
leak = [0.1, 0.3, 0.6, 0.9];
for m = 3:length(radii)
    m
    for l = 1:length(Nsize)
        for k = 1:N_trials
            net = Network();
            lID(1) = net.addLayer(1,                    'bias',     struct('nodeType', 'linear', 'leakageInit', 'constant', 'leakageVal', '1.0')); 
            lID(2) = net.addLayer(size(train_input, 1), 'input',    struct('nodeType', 'linear', 'leakageInit', 'constant', 'leakageVal', '0.0'));
            lID(3) = net.addLayer(size(train_output,1), 'output',   struct('nodeType', 'linear', 'leakageInit', 'constant', 'leakageVal', '1.0'));
            lID(4) = net.addLayer(Nsize(l),             'layer',    struct('nodeType', 'tanh',   'leakageInit', 'constant', 'leakageVal', num2str(leak(m))));
            net.setConnection(lID(2), lID(4), struct('initType', 'randn', 'connectivity', 1.0));
            %net.setConnection(lID(2), lID(3), struct('initType', 'randn', 'connectivity', 1.0, 'radius', 1.0));
            net.setConnection(lID(1), lID(3), struct('initType', 'randn', 'connectivity', 1.0));
            net.setConnection(lID(1), lID(4), struct('initType', 'randn', 'connectivity', 1.0));
            net.setConnection(lID(4), lID(3), struct('initType', 'randn', 'connectivity', 1.0));
            net.setConnection(lID(4), lID(4), struct('initType', 'rand', 'connectivity', 0.05, 'radius', 1.0));
            %net.visualize();

            boltrain = BatchOutputLayerTrain();
            initLen = 100;
            %tic
            x = boltrain.train(net, train_input, train_output, initLen, 'ridge1');
            %toc

            %tic
            net.rememberStates();
            y = net.predict(test_input);
            net.recallStates();
            %toc
            errors = y - test_output;
            sse005leak{m}(k, l) = sum(sum((y - test_output).^2))/length(y);
            disp(['MSE (' num2str(k) ', ' num2str(l) '): ', num2str(norm(y - test_output)^2/(length(y)))]);
        end
        figure, hold on;
        plot(test_output', 'linewidth', 2);
        plot(y','-.', 'linewidth', 2);
    end
end


rnnsse(:, 1) = [7.4092e-05 0.00011984 5.816e-05 0.00034397 0.00011471 6.2967e-05 0.0012148 8.6145e-05 0.00010519 5.4664e-05];
rnnsse(:, 2) = [0.0013756, 7.9694e-05, 0.00016942, 0.00016872, 0.00024495, 0.0001595, 0.00015273, 5.6336e-05, 0.00011722, 0.00010025];
rnnsse(:, 3) = [1.1125e-05, 7.3056e-05, 1.1903e-05, 1.1203e-05, 5.3446e-06, 5.521e-06, 1.1377e-05, 8.7959e-05, 1.2877e-05, 1.5512e-05];
rnnsse(:, 4) = [0.0042561, 0.0016267, 0.0025598, 0.0007366, 0.0014373, 0.0037354, 0.0063782, 0.0013964, 0.0026, 0.0010955];
rnnsse(:, 5) = [0.68146, 0.68058, 0.68793, 0.67961, 0.68935, 0.6945, 0.68249, 0.67781, 0.68683, 0.69582];
rnnsse(:, 6) = [0.72607, 0.69116, 0.74801, 0.68368, 0.69994, 0.72878, 0.69356, 0.7398, 0.72517, 0.78757];

% Elman
rnnsse(:, 3) = [3.5309e-05, 3.1988e-05, 3.736e-05, 3.8605e-05, 3.7624e-05, 3.6908e-05, 3.9798e-05, 3.8988e-05, 3.4671e-05, 3.5262e-05];
rnnsse(:, 4) = [0.0013915, 5.2053e-05, 0.00045596, 4.4878e-05, 0.00010384, 0.0011321, 0.00043468, 0.00054294, 0.00010454, 0.00017406];
rnnsse(:, 5) = [2.7607e-05, 1.7376e-05, 2.5608e-05, 2.7643e-05, 2.0126e-05, 3.1713e-05, 2.6281e-05, 1.506e-05, 2.6857e-05, 2.4815e-05];
rnnsse(:, 6) = [3.1426e-05, 1.3636e-05, 0.00017577, 3.8764e-05, 1.5015e-05, 1.2383e-05, 1.3637e-05, 1.303e-05, 1.3753e-05, 1.771e-05];
sse500_ = [3.1426e-05, 1.3636e-05, 3.8764e-05, 1.5015e-05, 1.2383e-05, 1.3637e-05, 1.303e-05, 1.3753e-05, 1.771e-05];

mElmanSSE = [0.00022345   0.00026244   2.4588e-05    0.00044366   2.4309e-05  1.8817e-05];
sElmanSSE = [0.00035848   0.0003948    3.42e-06      0.00047117   5.1666e-06  9.5312e-06];



combinedMeanSSE(:, 1) = mean(sse001);
combinedMeanSSE(:, 2) = mean(sse005);
combinedMeanSSE(:, 3) = mean(sse010);
combinedMeanSSE(:, 4) = mean(sse001_r075);
combinedMeanSSE(:, 5) = mean(sse005_r075);
combinedMeanSSE(:, 6) = mean(sse010_r075);
combinedMeanSSE(:, 7) = mean(rnnsse01);
combinedMeanSSE(:, 8) = mean(rnnsse001(1:9,:)');
combinedMeanSSE(:, 9) = mean(sse010_rad025);

combinedSTDSSE(:, 1) = std(sse001);
combinedSTDSSE(:, 2) = std(sse005);
combinedSTDSSE(:, 3) = std(sse010);
combinedSTDSSE(:, 4) = std(sse001_r075);
combinedSTDSSE(:, 5) = std(sse005_r075);
combinedSTDSSE(:, 6) = std(sse010_r075);
combinedSTDSSE(:, 7) = std(rnnsse01);
combinedSTDSSE(:, 8) = std(rnnsse001(1:9,:)')
combinedSTDSSE(:, 9) = std(sse010_rad025);

rnnsse001(1:9,:)

combinedMeanSSE(:, 4) = mean(rnnsse)';
combinedSTDSSE(:, 4) = std(rnnsse)';


plotPredictionAccuracy(combinedMeanSSE, combinedSTDSSE, [20,63,147,263,411,591,803,1047,1323],...
    {'ESN, con = 0.01, rad = 1.25', 'ESN, con = 0.05, rad = 1.25', 'ESN, con = 0.10, rad = 1.25',...
     'ESN, con = 0.01, rad = 0.75', 'ESN, con = 0.05, rad = 0.75', 'ESN, con = 0.10, rad = 0.75',...
    'BP RNN, alpha = 0.10', 'BP RNN, alpha = 0.01', 'ESN, con = 0.05, rad = 0.25'}, 1, struct('linewidth', 2))

dataLenght = size(train_input,2);
for l = 1:6
    paramNumber = size(Lorenz_data,1) * Nsize(l)  + size(Lorenz_data,1);
    correction = 0; % 2 * paramNumber * (paramNumber + 1) / (dataLenght - paramNumber - 1);
    akaikeESN001(l) = dataLenght *  log(mean(sse(:, l))) + 2 * paramNumber + correction;
end


for l = 1:6
    paramNumber = Nsize(l)^2 * size(Lorenz_data,1) * Nsize(l) + Nsize(l) + 3;
    correction = 0; %2 * paramNumber * (paramNumber + 1) / (dataLenght - paramNumber - 1);
    akaikeELMAN(l) = dataLenght *  log(mean(rnnsse(:, l))) + 2 * paramNumber + correction;
end

% ESN

% N = N_res * N_out + N_res + N_out;
% N = N_in * N_hid + N_hid * N_hid + N_hid * N_out + N_hid + N_out;
% 
% N_in * N_hid + N_hid * N_hid + N_hid * N_out =  N_res * N_out
% 
% (1/N_out)*N_hid^2 + (N_in/N_out + 1) * N_hid - N_res = 0;
% 
% D = (N_in/N_out + 1)^2 + 4 * N_res /(N_out);
% 
% x = - (N_in/N_out + 1) +/- sqrt(D) / (2 / N_out );

calcNnumWeightsInESN = @(N_hid)  N_hid * 3 + N_hid + 3

numNodesInReservoir = @(N_weights) (N_weights - 3)/4;

calcNnumWeightsInElman = @(N_hid) calcNnumWeightsInESN(N_hid) + 3 * N_hid + N_hid^2 ;

%numNodesInHidden = @(N_weights) (N_weights - calcNnumWeightsInESN(N_hid))

N = 33;
calcNnumWeightsInESN(10*N)
calcNnumWeightsInElman(N)

%Weights       20     63      147     263     411     591       803     1047       1323
%ESN   Nodes:   4     15       36      65     102     147       200      261       330            
%ELMAN Nodes:   2      5        9      13      17      21        25       29        33



%4
rnnsse(:, 1) = [0.00073801, 0.00073263, 0.00053681, 0.00053645, 0.00063376, 0.00043638, 0.00043186, 0.00084469, 0.00043516, 0.00086278, 0.00039689, 0.00075602, 0.00066295, 0.00045435, 0.00055188, 0.00045138, 0.00071571, 0.00064105, 0.00051523, 0.00039351];
% 15
rnnsse(:, 2) = [0.00013232, 0.00018142, 0.0001326, 0.00010135, 0.0001373, 0.00012293, 6.8166e-05, 6.5604e-05, 0.00011461, 7.4065e-05, 0.0001836, 0.00014843, 0.00015764, 0.0001774, 0.00013211, 0.00011449, 0.00012534, 0.00011882, 0.00017704, 0.00011725];
% 36
rnnsse(:, 3) = [7.2405e-05, 0.00038304, 0.00023522, 0.00025139, 0.00045882, 0.00010567, 9.2831e-05, 0.00016347, 0.00029842, 0.00020296, 0.00018163, 0.00024697, 9.2515e-05, 0.00012783, 0.00010111, 0.0002901, 0.00020067, 0.00016316, 0.00037999, 0.00019716];
% 65
rnnsse(:, 4) = [0.00022853, 0.00010164, 0.00059181, 0.00027031, 0.00028452, 0.00012878, 0.00029268, 0.00033524, 0.0032185, 0.00013421, 0.00045391, 0.0004941, 0.00013905, 0.00040899, 0.0003864, 0.00024511, 0.00034215, 0.00052851, 8.313e-05, 0.0002348];
% 0.001051

rnnsse01(:, 1) = [0.031012, 0.030999, 0.031219, 0.031024, 0.031065, 0.03101, 0.031026, 0.031161, 0.031015, 0.031229];
rnnsse01(:, 2) = [0.00017344, 7.6695e-05, 6.2131e-05, 6.5973e-05, 5.6947e-05, 8.1314e-05, 5.3735e-05, 9.2721e-05, 7.3382e-05, 7.3818e-05];
rnnsse01(:, 3) = [3.9684e-05, 4.3043e-05, 3.8146e-05, 4.1306e-05, 5.1202e-05, 3.7886e-05, 5.6901e-05, 4.6416e-05, 6.0003e-05, 3.5887e-05];
rnnsse01(:, 4) = [3.4013e-05, 3.8766e-05, 3.9869e-05, 3.7471e-05, 3.4031e-05, 3.8469e-05, 4.148e-05, 4.3366e-05, 4.3787e-05, 2.9429e-05]
rnnsse01(:, 5) = [2.204e-05, 4.2764e-05, 2.1935e-05, 2.8784e-05, 3.7669e-05, 3.8557e-05, 2.4366e-05, 2.8586e-05, 2.9994e-05, 1.978e-05];
rnnsse01(:, 6) = [2.4427e-05, 1.6648e-05, 1.241e-05, 9.884e-06, 1.7809e-05, 2.0239e-05, 1.4234e-05, 2.3962e-05, 1.7045e-05, 1.9752e-05];
rnnsse01(:, 7) = [1.6353e-05, 1.4785e-05, 2.0807e-05, 1.5816e-05, 1.4838e-05, 1.2194e-05, 2.5246e-05, 2.8493e-05, 2.7517e-05, 2.0787e-05];
rnnsse01(:, 8) = [1.0584e-05, 2.0596e-05, 1.6836e-05, 2.0687e-05, 1.7483e-05, 2.2636e-05, 1.0325e-05, 1.2622e-05, 2.2804e-05, 1.6422e-05];
rnnsse01(:, 9) = [1.4433e-05, 1.3169e-05, 1.2497e-05, 1.9162e-05, 1.5691e-05, 1.8125e-05, 1.3933e-05, 1.6067e-05, 1.179e-05, 1.3179e-05];



