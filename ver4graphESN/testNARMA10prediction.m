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
net = Network();

lID(1) = net.addLayer(1, 'bias',   struct('nodeType', 'linear', 'leakage', 1, 'initType', 'randn')); % All layers have bias
lID(2) = net.addLayer(1, 'input',  struct('nodeType', 'linear', 'leakage', 1.0));
lID(3) = net.addLayer(1, 'output', struct('nodeType', 'linear', 'leakage', 1.0));
lID(4) = net.addLayer(100, 'reservoir', struct('nodeType', 'tanh', 'radius', 1.0, 'leakage', 1.0, 'connectivity',1.0, 'initType', 'randn'));
lID(5) = net.addLayer(100, 'reservoir', struct('nodeType', 'tanh', 'radius', 1.0, 'leakage', 1.0, 'connectivity',1.0, 'initType', 'randn'));

connections = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]; % remove to last numbers ->  1.0, 1.0
arch = sparse([lID(2) lID(2) lID(1) lID(1) lID(1) lID(4) lID(5)],... %  remove to last IDS ->  lID(4) lID(5)
              [lID(4) lID(5) lID(3) lID(4) lID(5) lID(3) lID(3)], connections,6,6); % remove to last IDS ->  lID(3) lID(3)
net.setConnections(arch, 'randn');

% And uncomment the following lines
% lID(6) = net.addLayer(100, 'reservoir', struct('nodeType', 'tanh', 'radius', 1.0, 'leakage', 1.0, 'connectivity',1.0, 'initType', 'randn'));
% net.setConnection(lID(4), lID(6), struct('type', 'randn', 'connectivity', 0.01));
% net.setConnection(lID(5), lID(6), struct('type', 'randn', 'connectivity', 0.01));
% net.setConnection(lID(6), lID(3), struct('type', 'randn', 'connectivity', 1.0));

net.visualize();

%% Train the network
boltrain = BatchOutputLayerTrain();
initLen = 100;
tic
[~, x] = boltrain.train(net, train_input, train_output, initLen);
toc
% figure, plot(x')

%% Predict
tic
net.rememberStates();
y = net.predict(test_input);
net.recallStates();
toc
disp(['RMSE: ', num2str(sqrt(mse(y - test_output)))]);
disp(['NRMSE: ', num2str(NRMSE(y, test_output))]);

figure, hold on;
plot(test_output, 'linewidth', 2);
plot(y,'-.', 'linewidth', 2);
