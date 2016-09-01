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
lID(1) = net.addLayer(1, 'input', struct('node', 'linear'));
lID(2) = net.addLayer(1, 'output', struct('node', 'linear'));
lID(3) = net.addLayer(2000, 'reservoir', struct('node', 'tanh', 'radius', 0.8,'leakage', 0.2, 'connectivity',0.1, 'init_type', 'randn'));
lID(4) = net.addLayer(2000, 'reservoir', struct('node', 'tanh', 'radius', 0.2,'leakage', 0.2, 'connectivity',0.1, 'init_type', 'randn'));
lID(5) = net.addLayer(2000, 'reservoir', struct('node', 'tanh', 'radius', 0.8,'leakage', 0.2, 'connectivity',0.1, 'init_type', 'randn'));

connections = [1.0, 1.0, 0.1, 0.1, 0.1, 0.1];
arch = sparse([lID(1) lID(5) lID(3) lID(4) lID(2) lID(3)],...
              [lID(3) lID(2) lID(4) lID(5) lID(3) lID(5)], connections,5,5);
net.setConnections(arch, 'randn');
net.removeLayer(lID(4));
%net.visualize();

%% Testing the network
tic
for k = 1 : length(train_input)
    net.forward(train_input(k));
    y(k) = net.getOutput();
end
toc
figure, plot(y)


