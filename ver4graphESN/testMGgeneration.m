%% Notes
% (2) add noise paramter to layers' params
% (3) add withend reservoir 
% (4) rewrite sprand and sprandn
% (7) analyze ESN from Graph theoretic point of view i.e. how the length of
% shortest path from any node effects complexity of the generated signal


%% Training data preparation
addpath('../T0_chaotic/');
mackeyglass_data = mackeyglass(5000)';
p = 0.80;
mid_point = round(p * length(mackeyglass_data)); % take a small portion for training
train_input = 1*mackeyglass_data(:, 1:mid_point);
train_output = 1*mackeyglass_data(:, 2:mid_point + 1);
test_input = 1*mackeyglass_data(:, mid_point + 1:end - 1);
test_output = 1*mackeyglass_data(:, mid_point + 2:end);

%% Construct the network
net = Network();

lID(1) = net.addLayer(1, 'bias',   struct('nodeType', 'linear', 'leakage', 1, 'initType', 'randn')); % All layers have bias
lID(2) = net.addLayer(size(train_input,1), 'input',  struct('nodeType', 'linear', 'leakage', 1.0));
lID(3) = net.addLayer(size(train_output,1), 'output', struct('nodeType', 'linear', 'leakage', 1.0));
lID(4) = net.addLayer(1000, 'reservoir', struct('nodeType', 'tanh', 'radius', 1.0, 'leakage', 0.2, 'connectivity',0.1, 'initType', 'randn'));
% lID(5) = net.addLayer(50, 'reservoir', struct('nodeType', 'tanh', 'radius', 1.0, 'leakage', 0.6, 'connectivity',0.1, 'initType', 'randn'));
% lID(6) = net.addLayer(50, 'reservoir', struct('nodeType', 'tanh', 'radius', 1.0, 'leakage', 0.2, 'connectivity',0.1, 'initType', 'randn'));

connections = [1.0 1.0, 1.0, 1.0, ]; %1.0, 1.0, 1, 0.1, 0.1, 1
arch = sparse([ lID(2) lID(1) lID(1) lID(4)],... % lID(1) lID(5)  lID(4) lID(5) lID(6) lID(1)
              [ lID(4) lID(3) lID(4) lID(3) ], connections,6,6); % lID(5) lID(3)  lID(5) lID(6) lID(3) lID(6)
net.setConnections(arch, 'rand');
% net.visualize();


%% Test removal and addition of layers and connections
% net.removeLayer(lID(5));
% lID(5) = net.addLayer(100, 'reservoir', struct('nodeType', 'tanh', 'radius', 0.6, 'leakage', 0.2, 'connectivity',0.4, 'initType', 'randn'));
% net.setConnection(lID(1), lID(5), struct('type', 'randn', 'connectivity', 1.0));
% net.setConnection(lID(4), lID(5), struct('type', 'randn', 'connectivity', 0.5));
% net.setConnection(lID(5), lID(6), struct('type', 'randn', 'connectivity', 0.1));

%% Train the network
boltrain = BatchOutputLayerTrain();
initLen = 100;
tic
x = boltrain.train(net, train_input, train_output, initLen);
toc

% figure, plot(x')
%% Generate
tic
net.rememberStates();
y = net.generate(test_input(1), length(test_input));
net.recallStates();
toc

disp(['RMSE: ', num2str(norm(y - test_output)/sqrt(length(y)))]);
%disp(['NRMSE: ', num2str(NRMSE(y, test_output))]);

figure, hold on;
plot(test_output, 'linewidth', 2);
plot(y,'-.', 'linewidth', 2);
