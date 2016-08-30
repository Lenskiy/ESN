%% Multilayer perceptron for comparison
train_output_ = ind2vec(train_output_ext + 1);
train_input_  = train_input_ext/255;
maxPerImage = max(train_input_ext);
for k = 1:size(train_input_ext,2)
    train_input_(:,k) = train_input_ext(:,k)./maxPerImage(k);
end
test_output_ = ind2vec(test_output + 1);
%test_input_  = test_input/255;

net = feedforwardnet(300);
net.layers{1}.transferFcn = 'logsig';
net = configure(net,train_input_,train_output_);
net.trainFcn = 'trainscg';
%view(net);
%net.performFcn = 'sse';  % set performance functions that is used to evaluate 
net.divideParam.trainRatio = 0.8; % training set [%] 
net.divideParam.valRatio = 0.1; % validation set [%] 
net.divideParam.testRatio = 0.1; % test set [%] 
net.trainParam.goal = 0;
net.trainParam.epochs = 10000;
[net,tr,Y,E] = train(net,train_input_,full(train_output_), 'useParallel','yes', 'showResources','yes');

outputs = net(test_input_);
[val ind]= max(outputs);
errors = test_output - (ind - 1);
disp(['Error rate: ' num2str(1 - length(find(errors == 0))/size(test_output,2))]);

%trainoss, Error rate: 0.049217 
%trainrp,  Error rate: 0.04505
%traingdx, Error rate: 0.11153
%traingda, Error rate: 0.18745
%trainscg, Error rate: 0.04533

%Ext training set
%trainrp, Error rate: 0.0715
%trainscg, Error rate:0.0433