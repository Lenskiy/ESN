%% Load MNIST dataset
addpath('../T2_MNIST/');

mnisttrain = importdata('../T2_MNIST/mnist_train.csv',',');
mnisttest = importdata('../T2_MNIST/mnist_test.csv', ',');
selected_inds = randperm(size(train_input,2));
selected_inds = selected_inds(1:10000);

train_input =  mnisttrain(selected_inds,2:end)';
train_input_org = reshape(train_input, [28, 28, size(train_input,2)]);
train_output = ind2vec(mnisttrain(selected_inds,1)' + 1);
test_input = mnisttest(:,2:end)';
test_input_org = reshape(test_input, [28, 28, size(test_input,2)]);
test_output = ind2vec(mnisttest(:,1)' + 1);

%rotate
% for k = 1:size(train_input_org,3)
%     train_input_rot(:, :, k) = train_input_org(:, :, k)';
%     train_input_comb(:,:,k) = [train_input_org(:, :, k); train_input_rot(:, :, k)];
% end
% 
% for k = 1:size(test_input_org,3)
%     test_input_rot(:, :, k) = test_input_org(:, :, k)';
%     test_input_comb(:,:,k) = [test_input_org(:, :, k); test_input_rot(:, :, k)];
% end

%% Normalize
train_input_norm = train_input / 255;
test_input_norm = test_input / 255;

% figure, hold on;
% subplot(2,1,1), imshow(uint8(255*test_input_comb(:,:,1)));
% trImgs = uint8(generateProjections(img, 15, 0, 0));
% subplot(2,1,2), imshow(round(trImgs));
% hold off;

%% Construct the network
net = Network();

lID(1) = net.addLayer(1,                    'bias',     struct('nodeType', 'linear', 'leakageInit', 'constant', 'leakageVal', '1.0')); 
lID(2) = net.addLayer(size(train_input_norm, 1), 'input',    struct('nodeType', 'linear', 'leakageInit', 'constant', 'leakageVal', '0.0'));
lID(3) = net.addLayer(size(train_output,1), 'output',   struct('nodeType', 'linear', 'leakageInit', 'constant', 'leakageVal', '1.0'));

connections = [1.0, 1.0];
arch = sparse([lID(1) lID(2)],...
              [lID(3) lID(3)], connections, max(lID), max(lID));
initTypes = {'randn', 'randn', 'randn'};
radii = [1.0 1.0];
net.setConnections(arch, initTypes, radii);

net.visualize();
 
%% Train the network
boltrain = BatchOutputLayerTrain();
initLen = 0;
tic
error = boltrain.train(net, train_input_norm, train_output, initLen, 'sgd');
toc

%% Test
input = test_input_norm;
target = test_output;
indTarget = vec2ind(target);

nExamples = size(input,2);
   
y = net.predict(input);
[~, estClass] = max(y);
error = (nExamples - sum(estClass == indTarget))/nExamples;

