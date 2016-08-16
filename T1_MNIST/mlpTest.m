%% Multilayer perceptron for comparison
train_output_ = ind2vec(train_output + 1);
train_input_  = train_input/255;
net = feedforwardnet(300);
net.layers{1}.transferFcn = 'logsig';
net = configure(net,train_input_,train_output_);
net.trainFcn = 'trainoss';
%view(net);
% net.performFcn = 'sse';  % set performance functions that is used to evaluate 
% net.divideParam.trainRatio = 0.8; % training set [%] 
% net.divideParam.valRatio = 0.1; % validation set [%] 
% net.divideParam.testRatio = 0.1; % test set [%] 
% net.trainParam.goal = 0.001;
[net,tr,Y,E] = train(net,train_input_,full(train_output_), 'useParallel','yes', 'showResources','yes');

outputs = net(train_input_);
[val ind]= max(outputs);
errors = train_output - (ind - 1);
disp(['Error rate: ' num2str(1 - length(find(errors == 0))/size(train_output,2))]);

