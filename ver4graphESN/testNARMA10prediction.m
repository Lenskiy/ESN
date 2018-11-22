%% Training data preparation
addpath('../T1_NARMA/');
data = NARMA10series(2100);
testp = 0.5;
split_point = round((1-testp) * length(data)); % take testp as hold out test set
train_input  = data(1, 1:split_point,1);
size(train_input)
train_output = data(2, 2:split_point + 1);
test_input   = data(1, split_point + 1:end - 1);
test_output  = data(2, split_point + 2:end);


%% Construct the network
% net = Network();
% 
% lID(1) = net.addLayer(1, 'bias',   struct('nodeType', 'linear', 'leakage', 1, 'initType', 'randn')); % All layers have bias
% lID(2) = net.addLayer(size(train_input,1), 'input',  struct('nodeType', 'linear', 'leakage', 1.0));
% lID(3) = net.addLayer(size(train_output,1), 'output', struct('nodeType', 'linear', 'leakage', 1.0));
% lID(4) = net.addLayer(1000, 'reservoir', struct('nodeType', 'tanh', 'radius', 1.0, 'leakage', 1.0, 'connectivity',1.0, 'initType', 'randn'));
% lID(5) = net.addLayer(100, 'reservoir', struct('nodeType', 'tanh', 'radius', 1.0, 'leakage', 1.0, 'connectivity',1.0, 'initType', 'randn'));
% 
% connections = [1.0, 1.0, 1.0, 1.0, 1.0 1.0, 1.0]; % remove two last numbers ->  1.0, 1.0
% arch = sparse([lID(2) lID(2) lID(1) lID(1) lID(1) lID(4) lID(5)],... %  remove two last IDS ->  lID(4) lID(5)
%               [lID(4) lID(5) lID(3) lID(4) lID(5) lID(3) lID(3)], connections,5,5); % remove two last IDS ->  lID(3) lID(3)
% net.setConnections(arch, 'randn');

net = Network();

lID(1) = net.addLayer(1,                    'bias',     struct('nodeType', 'linear', 'leakageInit', 'constant', 'leakageVal', '1.0')); 
lID(2) = net.addLayer(size(train_input, 1), 'input',    struct('nodeType', 'linear', 'leakageInit', 'constant', 'leakageVal', '0.0'));
lID(3) = net.addLayer(size(train_output,1), 'output',   struct('nodeType', 'linear', 'leakageInit', 'constant', 'leakageVal', '1.0'));
lID(4) = net.addLayer(100,                 'layer',     struct('nodeType', 'tanh',  'leakageInit', 'constant', 'leakageVal', '0.08:0.12'));

net.setConnection(lID(2), lID(4), struct('initType', 'randn', 'connectivity', 1.0, 'radius', 1.0));
net.setConnection(lID(2), lID(3), struct('initType', 'randn', 'connectivity', 1.0, 'radius', 1.0));
net.setConnection(lID(1), lID(3), struct('initType', 'randn', 'connectivity', 1.0, 'radius', 1.0));
net.setConnection(lID(1), lID(4), struct('initType', 'randn', 'connectivity', 1.0, 'radius', 1.0));
net.setConnection(lID(4), lID(3), struct('initType', 'randn', 'connectivity', 1.0, 'radius', 1.0));
net.setConnection(lID(4), lID(4), struct('initType', 'randn', 'connectivity', 0.01, 'radius', 1.5));



%% Train the network
boltrain = BatchOutputLayerTrain();
initLen = 100;
tic
x = boltrain.train(net, train_input, train_output, initLen, 'ridge1');
toc 
% figure, plot(x')

%% Predict
tic
net.rememberStates();
y = net.predict(test_input);
net.recallStates();
toc 

disp(['RMSE: ', num2str(norm(y - test_output)/sqrt(length(y)))]);
disp(['NRMSE: ', num2str(NRMSE(y, test_output))]);

figure, hold on;
plot(test_output, 'linewidth', 2);
plot(y,'-.', 'linewidth', 2);
