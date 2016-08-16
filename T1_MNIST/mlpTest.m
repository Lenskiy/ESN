%% Multilayer perceptron for comparison
train_output_ = ind2vec(train_output' + 1);
net = feedforwardnet(10);
net = configure(net,train_input',train_output_);
view(net);
% net.performFcn = 'sse';  % set performance functions that is used to evaluate 
% net.divideParam.trainRatio = 0.8; % training set [%] 
% net.divideParam.valRatio = 0.1; % validation set [%] 
% net.divideParam.testRatio = 0.1; % test set [%] 
% net.trainParam.goal = 0.001;
[net,tr,Y,E] = train(net,train_input',train_output_, 'useParallel','yes', 'showResources','yes');

outputs = net(train_input');
[val ind]= max(outputs);
errors = gsubtract(train_output',ind - 1);
disp(['Error rate: ' num2str(1 - length(find(train_output' == 0))/size(train_output,1))]);

