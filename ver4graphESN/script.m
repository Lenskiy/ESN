%% Training data preparation
% make sure mackeyglassm is in the path
mackeyglass_data = mackeyglass(3100)';
mackeyglass_data = mackeyglass_data(101:2100);
mid_point = round(15 * length(mackeyglass_data) / 20); % take a small portion for training
train_input = mackeyglass_data(:, 1:mid_point);
train_output = mackeyglass_data(:, 2:mid_point + 1);
test_input = mackeyglass_data(:, mid_point + 1:end - 1);
test_output = mackeyglass_data(:, mid_point + 2:end);

%% Building network
net = Network();
net.addLayer(1,  1, 'input',  '@(x) x', []);
net.addLayer(2,  1, 'output', '@(x) x', []);
net.addLayer(3, 200, 'reservoir', '@(x) tanh(x)', struct('radius', 0.8,'leakage', 0.2, 'connectivity',0.4, 'init_type', 'randn'));
net.addLayer(4, 200, 'reservoir', '@(x) tanh(x)', struct('radius', 0.2,'leakage', 0.2, 'connectivity',0.1, 'init_type', 'randn'));
net.addLayer(5, 200, 'reservoir', '@(x) tanh(x)', struct('radius', 0.8,'leakage', 0.2, 'connectivity',0.6, 'init_type', 'randn'));
connections = [1.0, 1.0, 0.1, 0.1, 0.1, 0.1];
arch = sparse([1 5 3 4 2 3],...
              [3 2 4 5 3 5], connections,5,5);
net.setConnections(arch, 'randn');
net.visualize();

%% Testing the network

for k = 1 : length(train_input)
    net.forward(train_input(k));
    y(k) = net.getOutput();
end
figure, plot(y)


