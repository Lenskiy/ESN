%% Load MNIST dataset
addpath('../T2_MNIST/');

mnisttrain = importdata('../T2_MNIST/mnist_train.csv',',');
mnisttest = importdata('../T2_MNIST/mnist_test.csv', ',');
train_input =  mnisttrain(:,2:end)';
train_input_org = reshape(train_input, [28, 28, size(train_input,2)]);
train_output = ind2vec(mnisttrain(:,1)' + 1);
test_input = mnisttest(:,2:end)';
test_input_org = reshape(test_input, [28, 28, size(test_input,2)]);
test_output = ind2vec(mnisttest(:,1)' + 1);

%rotate
for k = 1:size(train_input_org,3)
    train_input_rot(:, :, k) = train_input_org(:, :, k)';
    train_input_comb(:,:,k) = [train_input_org(:, :, k); train_input_rot(:, :, k)];
end

for k = 1:size(test_input_org,3)
    test_input_rot(:, :, k) = test_input_org(:, :, k)';
    test_input_comb(:,:,k) = [test_input_org(:, :, k); test_input_rot(:, :, k)];
end

%% Normalize
train_input_comb = train_input_comb / 255;
test_input_comb = test_input_comb / 255;

% figure, hold on;
% subplot(2,1,1), imshow(uint8(255*test_input_comb(:,:,1)));
% trImgs = uint8(generateProjections(img, 15, 0, 0));
% subplot(2,1,2), imshow(round(trImgs));
% hold off;

%% Construct the network
net = Network();

lID(1) = net.addLayer(1, 'bias',   struct('nodeType', 'linear', 'leakage', 1, 'initType', 'randn')); % All layers have bias
lID(2) = net.addLayer(size(train_input_org,1), 'input',  struct('nodeType', 'linear', 'leakage', 1.0));
lID(3) = net.addLayer(size(train_input_rot,1), 'input',  struct('nodeType', 'linear', 'leakage', 1.0));
lID(4) = net.addLayer(size(train_output,1), 'output',  struct('nodeType', 'linear', 'leakage', 1.0));
lID(5) = net.addLayer(1000, 'reservoir', struct('nodeType', 'tanh', 'radius', 1.0, 'leakage', 1.0, 'connectivity',0.01, 'initType', 'randn'));
% lID(6) = net.addLayer(400, 'reservoir', struct('nodeType', 'tanh', 'radius', 1.0, 'leakage', 1.0, 'connectivity',0.001, 'initType', 'randn'));
% lID(7) = net.addLayer(200, 'reservoir', struct('nodeType', 'tanh', 'radius', 1.0, 'leakage', 1.0, 'connectivity',0.001, 'initType', 'randn'));


connections = [1.0, 1.0, 1.0 1.0];
arch = sparse([lID(1) lID(2) lID(3) lID(5)],...
              [lID(5) lID(5) lID(5) lID(4)], connections,5, 5);
net.setConnections(arch, 'randn');

net.visualize();
 
%% Train the network
boltrain = BatchTrainClassifierOutputLayer();
initLen = 0;
tic
error = boltrain.train(net, test_input_comb, test_output, initLen)
toc

%% Test
input = test_input_comb;
target = test_output;
indTarget = vec2ind(target);

nExamples = size(input,3);
nSamplesInExample = size(input,2);
            
outputId = net.getIdByName('output');
toOuputIDs = net.getPrevNodes(outputId);
                
avgStates = zeros(net.getNumberOfStates(toOuputIDs), nExamples);
            
W_out = net.getWeights(toOuputIDs, outputId);          
interestStates = zeros(net.getNumberOfStates(toOuputIDs), nSamplesInExample);
h = 0;   
tic
            for k = 1:nExamples
                if(mod(k, 100) == 0) 
                    k/nExamples
                end
                net.recallStates();
               
                for j = 1:nSamplesInExample
                    net.forward(input(:, j, k));
                    interestStates(:, j) = net.getStates(toOuputIDs);
                end          
                
                % Spatial average            
                avgStates(:, k) = mean(interestStates(:, initLen + 1:end),2);
            end
                
                P = W_out'*avgStates;
                
            for k = 1:nExamples    
                 [~, maxind] = max(P(:,k));  
                
                if maxind == indTarget(k)
                    h = h +1;
                end
                
            end
toc         
% Hit rate
error = 1 - h/nExamples
            

