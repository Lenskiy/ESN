%% Lorenz training/testing data
addpath('../T4_text_generation/');
fd = fopen('../T4_text_generation/input.txt', 'r');
input = fread(fd)';
fclose(fd);
%input = lower(input);
% input_std = std(input);
% input_mean = mean(input);
% input_n = (double(input) - input_mean)/input_std;
% train_input = input_n(1:end-1);
% train_output = input_n(2:end);

input_p = -1*ones(length(unique(input)), size(input,2));
charSet = unique(input);
for k = 1:size(input,2)
    input_p(input(k) == charSet, k) = 1;
end

train_input = input_p(:, 1:end-1);
train_output = input_p(:, 2:end);
%% Construct the network
net = Network();

lID(1) = net.addLayer(1,                    'bias',     struct('nodeType', 'linear', 'leakage', 1.0)); 
lID(2) = net.addLayer(size(train_input, 1), 'input',    struct('nodeType', 'linear', 'leakage', 1.0));
lID(3) = net.addLayer(size(train_output,1), 'output',   struct('nodeType', 'linear', 'leakage', 1.0));
lID(4) = net.addLayer(50,                  'layer',     struct('nodeType', 'tanh',   'leakage', 1.0));
lID(5) = net.addLayer(40,                  'layer',     struct('nodeType', 'tanh',   'leakage', 1.0));
lID(6) = net.addLayer(30,                  'layer',     struct('nodeType', 'tanh',   'leakage', 1.0));

% Add lateral connections i.e. make these layers reservoirs
net.setConnection(lID(4), lID(4), struct('initType', 'randn', 'connectivity', 0.1, 'radius', 1.0));
net.setConnection(lID(5), lID(5), struct('initType', 'randn', 'connectivity', 0.1, 'radius', 1.0));
net.setConnection(lID(6), lID(6), struct('initType', 'randn', 'connectivity', 0.1, 'radius', 1.0));
% Add bias to evey layer
net.setConnection(lID(1), lID(3), struct('initType', 'randn', 'connectivity', 1.0, 'radius', 1.0));
net.setConnection(lID(1), lID(4), struct('initType', 'randn', 'connectivity', 1.0, 'radius', 1.0));
net.setConnection(lID(1), lID(5), struct('initType', 'randn', 'connectivity', 1.0, 'radius', 1.0));
net.setConnection(lID(1), lID(6), struct('initType', 'randn', 'connectivity', 1.0, 'radius', 1.0));
% Connect every reservoir to  output
%net.setConnection(lID(4), lID(3), struct('initType', 'randn', 'connectivity', 1.0, 'radius', 1.0));
% net.setConnection(lID(5), lID(3), struct('initType', 'randn', 'connectivity', 1.0, 'radius', 1.0));
net.setConnection(lID(6), lID(3), struct('initType', 'randn', 'connectivity', 1.0, 'radius', 1.0));
% Connect input to output
net.setConnection(lID(2), lID(3), struct('initType', 'randn', 'connectivity', 0.1, 'radius', 1.0));
% Connect reservoirs in a feed-forward structure 
net.setConnection(lID(2), lID(4), struct('initType', 'randn', 'connectivity', 0.1, 'radius', 1.0));
net.setConnection(lID(4), lID(5), struct('initType', 'randn', 'connectivity', 0.1, 'radius', 1.0));
net.setConnection(lID(5), lID(6), struct('initType', 'randn', 'connectivity', 0.1, 'radius', 1.0));



% connections = [1.0, 1.0, 1.0, 1.0, 1.0, 0.1];
% radii = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
% initType = {'randn', 'randn', 'randn', 'randn', 'randn', 'randn'};
% arch = sparse([lID(2) lID(2) lID(1) lID(1) lID(4) lID(4)],...
%               [lID(4) lID(3) lID(3) lID(4) lID(3) lID(4)], connections, 6, 6);
          
%net.setConnections(arch, initType, radii);

% net.visualize();

%% Train the network
boltrain = BatchOutputLayerTrain();
net.setConnectivity(0.01, lID(4));
initLen = 1;
tic
x = boltrain.train(net, train_input, train_output, initLen);
toc
% figure, plot(x')

%% Generate
tic
net.rememberStates();
y = net.generate(train_input(:,1), length(train_input));
%y = net.predict(train_input);
net.recallStates();
toc

for k = 1:size(y, 2)
    [val, arg] = max(y(:,k));
    generatedText(k) = charSet(arg);
end
char(generatedText)
figure, plot(y');
%generatedText = char(y * input_std + input_mean);

