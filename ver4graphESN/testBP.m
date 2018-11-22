%% Lorenz training/testing data
addpath('../T0_chaotic/');
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

%% Construct the network
net = Network();

lID(1) = net.addLayer(1,                    'bias',     struct('nodeType', 'linear', 'leakageInit', 'constant', 'leakageVal', '1.0')); 
lID(2) = net.addLayer(size(train_input, 1), 'input',    struct('nodeType', 'linear', 'leakageInit', 'constant', 'leakageVal', '0.0'));
lID(3) = net.addLayer(size(train_output,1), 'output',   struct('nodeType', 'linear', 'leakageInit', 'constant', 'leakageVal', '1.0'));
lID(4) = net.addLayer(100,                  'layer',     struct('nodeType', 'tanh',   'leakageInit', 'constant', 'leakageVal', '0.05'));
lID(5) = net.addLayer(100,                  'layer',     struct('nodeType', 'tanh',   'leakageInit', 'constant', 'leakageVal', '0.05'));
%lID(5) = net.addLayer(20,                  'layer',     struct('nodeType', 'tanh',   'leakageInit', 'rand', 'leakageVal', '0.8:1.0'));
%lID(6) = net.addLayer(10,                  'layer',     struct('nodeType', 'tanh',   'leakageInit', 'rand', 'leakageVal', '0.8:1.0'));


net.setConnection(lID(2), lID(4), struct('initType', 'randn', 'connectivity', 1.0, 'radius', 1.0));
net.setConnection(lID(2), lID(3), struct('initType', 'randn', 'connectivity', 1.0, 'radius', 1.0));
net.setConnection(lID(1), lID(3), struct('initType', 'randn', 'connectivity', 1.0, 'radius', 1.0));
net.setConnection(lID(1), lID(4), struct('initType', 'randn', 'connectivity', 1.0, 'radius', 1.0));
net.setConnection(lID(4), lID(5), struct('initType', 'randn', 'connectivity', 1.0, 'radius', 1.0));
net.setConnection(lID(5), lID(3), struct('initType', 'randn', 'connectivity', 1.0, 'radius', 1.0));
%net.setConnection(lID(1), lID(5), struct('initType', 'randn', 'connectivity', 1.0, 'radius', 1.0));
%net.setConnection(lID(1), lID(6), struct('initType', 'randn', 'connectivity', 1.0, 'radius', 1.0));
net.setConnection(lID(4), lID(4), struct('initType', 'randn', 'connectivity', 0.1, 'radius', 0.1));
net.setConnection(lID(5), lID(5), struct('initType', 'randn', 'connectivity', 0.1, 'radius', 0.1));
%net.setConnection(lID(5), lID(5), struct('initType', 'randn', 'connectivity', 0.2, 'radius', 0.6));
%net.setConnection(lID(6), lID(6), struct('initType', 'randn', 'connectivity', 0.05,'radius', 0.5));

%net.setConnection(lID(4), lID(5), struct('initType', 'randn', 'connectivity', 0.2, 'radius', 1.0));
%net.setConnection(lID(5), lID(6), struct('initType', 'randn', 'connectivity', 0.1, 'radius', 1.0));
%net.setConnection(lID(6), lID(4), struct('initType', 'randn', 'connectivity', 0.1, 'radius',0.5));

% net.removeLayer(lID(4))
% net.setConnection(lID(2), lID(5), struct('initType', 'randn', 'connectivity', 1.0, 'radius', 1.0));
% net.setConnection(lID(6), lID(3), struct('initType', 'randn', 'connectivity', 1.0, 'radius', 1.0));

% net.removeLayer(lID(5))
% net.removeLayer(lID(6))

% connections = [1.0, 1.0, 1.0, 1.0  1.0 0.01 0.01 0.01];
% arch = sparse([lID(2) lID(2) lID(1) lID(1) lID(4) lID(4) lID(5) lID(6) ],...
%               [lID(4) lID(3) lID(3) lID(4) lID(3) lID(5) lID(6) lID(4)], connections,7,7);
% net.setConnections(arch, 'randn');

% net.visualize();

%% Train the network
boltrain = BPTrain();
initLen = 100;
tic
x = boltrain.train(net, train_input, train_output, initLen, 'ridge1');
toc 
% figure, plot(x')

%% Generate/Predict
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


