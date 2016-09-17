%% Mackeyglass training/testing data
mackeyglass_data = mackeyglass(3100)';
mackeyglass_data = mackeyglass_data(101:2100);
mid_point = round(15 * length(mackeyglass_data) / 20); % take a small portion for training
train_input = mackeyglass_data(:, 1:mid_point);
train_output = mackeyglass_data(:, 2:mid_point + 1);
test_input = mackeyglass_data(:, mid_point + 1:end - 1);
test_output = mackeyglass_data(:, mid_point + 2:end);
%% NARMA10
data = NARMA100LTdepOnly(5000);
testp = 0.2;
split_point = round((1-testp) * length(data)); % take testp as hold out test set
train_input  = data(1, 1:split_point,1);
size(train_input)
train_output = data(2, 2:split_point + 1);
test_input   = data(1, split_point + 1:end - 1);
test_output  = data(2, split_point + 2:end);


%% ESN TEST
architecture = struct('inputDim',   size(train_input,1), 'numNodes',   10, 'outputDim',  size(train_output,1));      
% parameters  = struct('node_type','tanh', 'radius', 0.8,'leakage', 0.2, 'connectivity',0.1, 'init_type', 'rand');  
% parameters  = struct('node_type','tanh', 'radius', 0.2,'leakage', 0.5, 'connectivity',0.1, 'init_type', 'rand');
% parameters  = struct('node_type','tanh', 'radius', 0.4,'leakage', 0.9, 'connectivity',0.1, 'init_type', 'rand');
parameters  = struct('node_type','tanh', 'radius', 0.8,'leakage', 0.2, 'connectivity',0.1, 'init_type', 'randn');  

esn = ESN(architecture, parameters);
train = RRTrain();

initL = 100;

[err, states] =  train.train(esn, train_input, train_output, initL);
Y = esn.generate(train_input(1, :), size(train_output(1,:),2), 1);
%Y = esn.predict(test_input(1, :), 1);

%NRMSE(Y(50:end),test_output(1,50:end))
figure, hold on; title('system');
plot(train_input(1,:));
plot(Y(1,:));


%% Stacked ESNs TEST
sArchitecture = struct('inputDim',  size(train_input,1), 'numNodes',   [100; 100; 100; 100], 'outputDim',  size(train_output,1));
                  
sParameters(1)  = struct('node_type','tanh', 'radius', 0.8,'leakage', 0.2, 'connectivity',0.1, 'init_type', 'rand');
sParameters(2)  = struct('node_type','tanh', 'radius', 0.8,'leakage', 0.2, 'connectivity',0.1, 'init_type', 'rand');
sParameters(3)  = struct('node_type','tanh', 'radius', 0.8,'leakage', 0.2, 'connectivity',0.1, 'init_type', 'rand');
sParameters(4)  = struct('node_type','tanh', 'radius', 0.3,'leakage', 0.9, 'connectivity',0.1, 'init_type', 'rand');

sESN = StackedESN(sArchitecture, sParameters, 'rand');
train = Train();
initL = 100;
train.train(sESN, train_input, train_output, initL)

Y = sESN.generate(test_input(1, :), size(test_output(1,:),2), 1);
%Y = sESN.predict(test_input(1, :), 1);

NRMSE(Y,test_output(1,:))

figure, hold on; title('NARMA10 test');
plot(test_output(1,:));
plot(Y(1,:));

%% Hierarhical ESNs TEST
sArchitecture = struct('topology', 'rand', 'inputDim',  size(train_input,1), 'numNodes',   [500; 500; 500; 500], 'outputDim',  size(train_output,1));
% Paramteres of the top reservoir                 
hParameters  = struct('radius', 0.3, 'leakage',     0.5, 'connectivity',0.05, 'init_type', 'rand');
% Paramteres of the reservoirs                
sParameters(1)  = struct('node_type','tanh', 'radius', 0.8,'leakage', 0.2, 'connectivity',0.2, 'init_type', 'rand');
sParameters(2)  = struct('node_type','tanh', 'radius', 0.2,'leakage', 0.5, 'connectivity',0.2, 'init_type', 'rand');
sParameters(3)  = struct('node_type','tanh', 'radius', 0.4,'leakage', 0.9, 'connectivity',0.2, 'init_type', 'rand');
sParameters(4)  = struct('node_type','tanh', 'radius', 0.3,'leakage', 0.9, 'connectivity',0.2, 'init_type', 'rand')
%sParameters(4)  = struct('node_type','tanh', 'radius', 0.8,'leakage', 0.9, 'connectivity',0.1, 'init_type', 'rand'); 
hESN = HESN(sArchitecture, hParameters, sParameters);
train = RRTrain();
[err, states]= train.train(hESN, train_input, train_output, 1);err

%Y = hESN.generate(test_input(1, :), size(test_output(1,:),2), 1);
Y = hESN.predict(test_input(1, :), 1);

NRMSE(Y(50:end),test_output(1,50:end))

figure, 
subplot(2,1,1), hold on
% axis([1,size(states,2), -1, 1])
% d1 = plot(1:size(states,2), states(1:sArchitecture.numNodes(1),:)','r');
% d2 = plot(1:size(states,2), states(sArchitecture.numNodes(1) + 1:sum(sArchitecture.numNodes(1:2)),:)','g');
% d3 = plot(1:size(states,2), states(sum(sArchitecture.numNodes(1:2)) + 1:sum(sArchitecture.numNodes(1:3)),:)','b');
% d4 = plot(1:size(states,2), states(sum(sArchitecture.numNodes(1:3)) + 1:end,:)','k');
% title('States during the learning stage')
% legend([d1(1), d2(1), d3(1), d4(1)],['\rho_1 = ', num2str(sParameters(1).radius), '  l_1 = ', num2str(sParameters(1).radius)],...
%                       ['\rho_2 = ', num2str(sParameters(2).radius), '  l_2 = ', num2str(sParameters(2).leakage)],...
%                       ['\rho_3 = ', num2str(sParameters(3).radius), '  l_3 = ', num2str(sParameters(3).leakage)],...
%                       ['\rho_4 = ', num2str(sParameters(4).radius), '  l_4 = ', num2str(sParameters(4).leakage)]);

subplot(2,1,2), hold on
title('system');
plot(test_output(1,:));
plot(Y(1,:));
