%% Manipluator trajectory
addpath('../T5_HCC/');
[hx, hy, dx, dy, cs, ~, ~] = textread('../T5_HCC/TJ_2_CCW.txt','%f%f%f%f%f%f%f','delimiter',' ');
input = [hx, hy]';
output= [dx, dy]';
input = input(:, 200:end);
output = output(:, 200:end);
output_compensate = [hx - dx, hy - dy]';
output_compensate = output_compensate(:, 200:end);
%input = [input input input input];
%output_compensate = [output_compensate output_compensate output_compensate output_compensate];
%output = [output output output output];
figure, hold on
plot(input(1, :), input(2, :));
plot(output(1, :), output(2, :));
plot(output_compensate(1, :), output_compensate(2, :));
legend({'human-input', 'desired-output', 'error to be predicted'});

p = 0.6;
mid_point = round((length(input) - 1) * p);
train_input = input(:, 1:mid_point);
train_output_compensate = output_compensate(:, 2:mid_point + 1);
test_input = input(:, mid_point + 1:end - 1);
test_output_compensate = output_compensate(:, mid_point + 2:end, :);
test_output = output(:, mid_point + 2:end, :);

%% Construct the network
net = Network();

lID(1) = net.addLayer(1,                    'bias',     struct('nodeType', 'linear', 'leakageInit', 'constant', 'leakageVal', '1.0')); 
lID(2) = net.addLayer(size(train_input, 1), 'input',    struct('nodeType', 'linear', 'leakageInit', 'constant', 'leakageVal', '0.0'));
lID(3) = net.addLayer(size(train_output_compensate,1), 'output',   struct('nodeType', 'linear', 'leakageInit', 'constant', 'leakageVal', '1.0'));
lID(4) = net.addLayer(1000,                  'layer',     struct('nodeType', 'tanh',  'leakageInit', 'constant', 'leakageVal', '0.08:0.12'));

net.setConnection(lID(2), lID(4), struct('initType', 'randn', 'connectivity', 1.0, 'radius', 1.0));
net.setConnection(lID(2), lID(3), struct('initType', 'randn', 'connectivity', 1.0, 'radius', 1.0));
net.setConnection(lID(1), lID(3), struct('initType', 'randn', 'connectivity', 1.0, 'radius', 1.0));
net.setConnection(lID(1), lID(4), struct('initType', 'randn', 'connectivity', 1.0, 'radius', 1.0));
net.setConnection(lID(4), lID(3), struct('initType', 'randn', 'connectivity', 1.0, 'radius', 1.0));
net.setConnection(lID(4), lID(4), struct('initType', 'randn', 'connectivity', 0.01, 'radius', 1.5));

% net.visualize();
%% Train the network
boltrain = BatchOutputLayerTrain();
initLen = 100;
tic
x = boltrain.train(net, train_input, train_output_compensate, initLen, 'ridge1');
toc 
figure, plot(x')

%% Test the network
tic
net.rememberStates();
gen_output = net.predict(test_input);
net.recallStates();
toc 

disp(['RMSE: ', num2str(norm(gen_output - test_output_compensate)/sqrt(length(gen_output)))]);

figure, hold on;
plot(test_input(1, :), test_input(2, :));
plot(test_output(1, :), test_output(2, :), 'linewidth', 2);
plot(test_input(1, :) - gen_output(1, :), test_input(2, :) - gen_output(2, :), '-.', 'linewidth', 2);
%plot(test_output(1, :) + gen_output(1, :), test_output(2, :) + gen_output(2, :), '-.', 'linewidth', 2);
xlabel('x(t)');
ylabel('y(t)');
legend({'human-input', 'desired-output', 'corrected by ESN'});

figure, hold on;
plot(test_output_compensate(1, :));
plot(gen_output(1, :));
title('compensation in x');
xlabel('t');
legend({'true', 'predicted'});

figure, hold on;
plot(test_output_compensate(2, :));
plot(gen_output(2, :));
title('compensation in y');
xlabel('t');
legend({'true', 'predicted'});
