%% Notes
% (2) add noise paramter to layers' params
% (3) add withend reservoir 
% (4) rewrite sprand and sprandn
% (6) add possibility to remove connections
% (7) analyze ESN from Graph theoretic point of view i.e. how the length of shortest path
% from any node effects complexity of the generated signal

%% Training data preparation
addpath('../T0_chaotic/');
mackeyglass_data = mackeyglass(3100)';
mackeyglass_data = mackeyglass_data(101:2100);
mid_point = round(15 * length(mackeyglass_data) / 20); % take a small portion for training
train_input = mackeyglass_data(:, 1:mid_point);
train_output = mackeyglass_data(:, 2:mid_point + 1);
test_input = mackeyglass_data(:, mid_point + 1:end - 1);
test_output = mackeyglass_data(:, mid_point + 2:end);

%% Construct the network
net = Network();

lID(1) = net.addLayer(1, 'bias',   struct('nodeType', 'linear', 'leakage', 1, 'initType', 'randn')); % All layers have bias
lID(2) = net.addLayer(1, 'input',  struct('nodeType', 'linear', 'leakage', 1.0));
lID(3) = net.addLayer(1, 'output', struct('nodeType', 'linear', 'leakage', 1.0));
lID(4) = net.addLayer(250, 'reservoir', struct('nodeType', 'tanh', 'radius', 1.0, 'leakage', 0.2, 'connectivity',0.2, 'initType', 'randn'));
lID(5) = net.addLayer(200, 'reservoir', struct('nodeType', 'tanh', 'radius', 0.6, 'leakage', 0.2, 'connectivity',0.4, 'initType', 'randn'));
lID(6) = net.addLayer(150, 'reservoir', struct('nodeType', 'tanh', 'radius', 0.3, 'leakage', 0.2, 'connectivity',0.8, 'initType', 'randn'));

connections = [1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.1, 1.0, 0.00, 0.2];
arch = sparse([lID(2) lID(1) lID(1) lID(1) lID(1) lID(4) lID(5) lID(6), lID(6), lID(4)],...
              [lID(4) lID(3) lID(4) lID(5) lID(6) lID(5) lID(6) lID(3), lID(4), lID(6)], connections,6,6);
net.setConnections(arch, 'randn');
% net.visualize();


%% Test removal and addition of layers and connections
net.removeLayer(lID(5));
lID(5) = net.addLayer(100, 'reservoir', struct('nodeType', 'tanh', 'radius', 0.6, 'leakage', 0.2, 'connectivity',0.4, 'initType', 'randn'));
net.setConnection(lID(1), lID(5), struct('type', 'randn', 'connectivity', 1.0));
net.setConnection(lID(4), lID(5), struct('type', 'randn', 'connectivity', 0.5));
net.setConnection(lID(5), lID(6), struct('type', 'randn', 'connectivity', 0.1));

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

disp(['RMSE: ', num2str(sqrt(mse(y - test_output)))]);
disp(['NRMSE: ', num2str(NRMSE(y, test_output))]);

figure, hold on;
plot(test_output, 'linewidth', 2);
plot(y,'-.', 'linewidth', 2);
