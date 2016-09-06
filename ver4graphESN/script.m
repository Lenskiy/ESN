%% Training data preparation
addpath('../T0_chaotic/');
mackeyglass_data = mackeyglass(3100)';
mackeyglass_data = mackeyglass_data(101:2100);
mid_point = round(15 * length(mackeyglass_data) / 20); % take a small portion for training
train_input = mackeyglass_data(:, 1:mid_point);
train_output = mackeyglass_data(:, 2:mid_point + 1);
test_input = mackeyglass_data(:, mid_point + 1:end - 1);
test_output = mackeyglass_data(:, mid_point + 2:end);

%% Building network
net = Network();
% All networks have bias
lID(1) = net.addLayer(1, 'bias',   struct('nodeType', 'linear', 'leakage', 1, 'initType', 'randn'));
lID(2) = net.addLayer(1, 'input',  struct('nodeType', 'linear', 'leakage', 1));
lID(3) = net.addLayer(1, 'output', struct('nodeType', 'linear', 'leakage', 0.2));
lID(4) = net.addLayer(2, 'reservoir', struct('nodeType', 'tanh', 'radius', 0.8, 'leakage', 0.2, 'connectivity',1.0, 'initType', 'randn'));
lID(5) = net.addLayer(2, 'reservoir', struct('nodeType', 'tanh', 'radius', 0.2, 'leakage', 0.2, 'connectivity',1.0, 'initType', 'randn'));
lID(6) = net.addLayer(3, 'reservoir', struct('nodeType', 'tanh', 'radius', 0.8, 'leakage', 0.2, 'connectivity',1.0, 'initType', 'randn'));
% (1) add a bais type of layer maybe make it default one
% (2) add noise paramter to layers' params
% (3) add withend reservoir 
% (4) rewrite sprand and sprandn
%net.changeLayerParamsTo(lID(4), struct('nodeType', 'tanh', 'radius', 0.3, 'leakage', 0.3));

connections = [1.0, 1.0, 0.1, 0.1, 0.1, 0.1, 1, 1, 1, 1];
arch = sparse([lID(2) lID(6) lID(4) lID(5) lID(3) lID(4) lID(1) lID(1) lID(1) lID(1)],...
              [lID(4) lID(3) lID(5) lID(6) lID(4) lID(6) lID(3) lID(4) lID(5) lID(6)], connections,6,6);
net.setConnections(arch, 'randn');
% net.removeLayer(lID(4));
% lID(4) = net.addLayer(2000, 'reservoir', struct('nodeType', 'tanh', 'radius', 0.2,'leakage', 0.2, 'connectivity',0.1, 'initType', 'randn'));
% net.setConnection(lID(3),lID(4), struct('type', 'randn', 'connectivity', 0.1));
% net.setConnection(lID(4),lID(5), struct('type', 'randn', 'connectivity', 0.1));
% net.removeLayer(lID(4));
%net.visualize();

%% Testing the network
tic
y = net.generate(train_input);
toc
figure, plot(y)

boltrain = BatchOutputLayerTrain();
initLen = 100;
boltrain.train(net, train_input, train_output, initLen);
