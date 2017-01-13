# Echo state networks

```
%% Lorenz training/testing data
addpath('../T0_chaotic/');
[x1, x2, x3] = lorenz(28, 10, 8/3);
dataLen = 10000;
startInd = 1;
Lorenz_data = [x1(startInd:startInd + dataLen),...
               x2(startInd:startInd + dataLen),...
               x3(startInd:startInd + dataLen)]';

p = 0.75;
mid_point = round((length(Lorenz_data) - 1) * p);
train_input = Lorenz_data(:, 1:mid_point);
train_output = Lorenz_data(:, 2:mid_point + 1);
test_input = Lorenz_data(:, mid_point + 1:end - 1);
test_output = Lorenz_data(:, mid_point + 2:end, :);

%% Construct the network
net = Network();

lID(1) = net.addLayer(1, 'bias',   struct('nodeType', 'linear', 'leakage', 1, 'initType', 'randn')); % All layers have bias
lID(2) = net.addLayer(size(train_input,1), 'input',  struct('nodeType', 'linear', 'leakage', 1.0));
lID(3) = net.addLayer(size(train_output,1), 'output', struct('nodeType', 'linear', 'leakage', 1.0));
lID(4) = net.addLayer(200, 'reservoir', struct('nodeType', 'tanh', 'radius', 1.0, 'leakage', 0.2, 'connectivity',0.25, 'initType', 'randn'));
lID(5) = net.addLayer(100, 'reservoir', struct('nodeType', 'tanh', 'radius', 0.6, 'leakage', 0.6, 'connectivity',0.15, 'initType', 'randn'));
lID(6) = net.addLayer(50, 'reservoir', struct('nodeType', 'tanh', 'radius', 0.5, 'leakage', 1.0, 'connectivity',0.1, 'initType', 'randn'));

connections = [1.0, 1.0, 1.0, 1.0  1.0 0.01 0.01 0.01];
arch = sparse([lID(2) lID(2) lID(1) lID(1) lID(4) lID(4) lID(5) lID(6) ],...
              [lID(4) lID(3) lID(3) lID(4) lID(3) lID(5) lID(6) lID(4)], connections,7,7);
net.setConnections(arch, 'randn');

% net.visualize();

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
y = net.generate(test_input);
net.recallStates();
toc

disp(['RMSE: ', num2str(norm(y - test_output)/sqrt(length(y)))]);
%disp(['NRMSE: ', num2str(NRMSE(y, test_output))]);

figure, hold on;
plot(test_output', 'linewidth', 2);
plot(y','-.', 'linewidth', 2);

figure, hold on;
plot3(test_output(1,:), test_output(2,:), test_output(3,:));
plot3(y(1,:), y(2,:), y(3,:));
```

![](/ver4graphESN/demo.png "Grapth based ESN")
