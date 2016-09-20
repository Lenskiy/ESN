%% Load MNIST dataset
addpath('../T2_MNIST/');

mnisttrain = importdata('../T2_MNIST/mnist_train.csv',',');
mnisttest = importdata('../T2_MNIST/mnist_test.csv', ',');
train_input =  mnisttrain(:,2:end)';
train_input = reshape(train_input, [28, 28, size(train_input,2)]);
train_output = ind2vec(mnisttrain(:,1)' + 1);
test_input = mnisttest(:,2:end)';
test_input = reshape(test_input, [28, 28, size(test_input,2)]);
test_output = ind2vec(mnisttest(:,1)' + 1);


% img = reshape(test_input(:,18), [28, 28])';
% figure, hold on;
% subplot(2,1,1), imshow(uint8(img));
% trImgs = uint8(generateProjections(img, 15, 0, 0));
% subplot(2,1,2), imshow(round(trImgs));
% hold off;

%% Construct the network
net = Network();

lID(1) = net.addLayer(1, 'bias',   struct('nodeType', 'linear', 'leakage', 1, 'initType', 'randn')); % All layers have bias
lID(2) = net.addLayer(size(train_input,1), 'input',  struct('nodeType', 'linear', 'leakage', 1.0));
lID(3) = net.addLayer(size(train_output,1), 'output', struct('nodeType', 'linear', 'leakage', 1.0));
lID(4) = net.addLayer(1000, 'reservoir', struct('nodeType', 'tanh', 'radius', 1.0, 'leakage', 0.2, 'connectivity',0.25, 'initType', 'randn'));


connections = [1.0, 1.0, 1.0, 1.0];
arch = sparse([lID(1) lID(1) lID(2) lID(4)],...
              [lID(3) lID(4) lID(4) lID(3)], connections,4,4);
net.setConnections(arch, 'randn');

net.visualize();
 
%% Train the network
boltrain = BatchTrainClassifierOutputLayer();
initLen = 0;
tic
error = boltrain.train(net, train_input, train_output, initLen);
toc

%% Test
nExamples = size(test_input,3);

indTarget = vec2ind(test_output);

h = 0;
tic
for k = 1:nExamples
    k/nExamples
    net.rememberStates();
	y = net.predict(test_input(:,:,k));
	net.recallStates();

    [~, maxind] = max(mean(y,2));
	if maxind == indTarget(k)
        h = h + 1;
	end
end
toc
         
% Hit rate
error = 1 - h/nExamples
            

