%% Text generation task
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
training_length = round(0.8*length(input));
input = uint16(lower(char(input)));
input_p = ones(length(unique(input)), length(input));
charSet = unique(input);
%charSet = uint16(unique(lower(char(charSet))));
clear input_p
for k = 1:size(input,2)
    input_p(input(k) == charSet, k) = 1;
end

truePDF = sum(input_p,2)/length(input_p);

train_input = input_p(:, 1:training_length-1);
train_output = input_p(:, 2:training_length);

test_input = input_p(:, training_length:end-1);
test_output = input_p(:, training_length+1:end);
%% Construct the network
net = Network();

lID(1) = net.addLayer(1,                    'bias',     struct('nodeType', 'linear', 'leakageInit', 'constant', 'leakageVal', '1.0')); 
lID(2) = net.addLayer(size(train_input, 1), 'input',    struct('nodeType', 'linear', 'leakageInit', 'constant', 'leakageVal', '0.0'));
lID(3) = net.addLayer(size(train_output,1), 'output',   struct('nodeType', 'linear', 'leakageInit', 'constant', 'leakageVal', '1.0'));
lID(4) = net.addLayer(1000,                  'layer',     struct('nodeType', 'tanh',   'leakageInit', 'rand', 'leakageVal', '0.95:1.0'));
% lID(5) = net.addLayer(200,                  'layer',     struct('nodeType', 'tanh',   'leakageInit', 'rand', 'leakageVal', '0.8:1.0'));
% lID(6) = net.addLayer(100,                  'layer',     struct('nodeType', 'tanh',   'leakageInit', 'rand', 'leakageVal', '0.8:1.0'));

% Add lateral connections i.e. make these layers reservoirs
net.setConnection(lID(4), lID(4), struct('initType', 'randn', 'connectivity', 0.05, 'radius', 0.01));
% net.setConnection(lID(5), lID(5), struct('initType', 'randn', 'connectivity', 0.1, 'radius', 1.0));
% net.setConnection(lID(6), lID(6), struct('initType', 'randn', 'connectivity', 0.1, 'radius', 1.0));
% Add bias to evey layer
net.setConnection(lID(1), lID(3), struct('initType', 'randn', 'connectivity', 1.0, 'radius', 1.0));
net.setConnection(lID(1), lID(4), struct('initType', 'randn', 'connectivity', 1.0, 'radius', 1.0));
% net.setConnection(lID(1), lID(5), struct('initType', 'randn', 'connectivity', 1.0, 'radius', 1.0));
% net.setConnection(lID(1), lID(6), struct('initType', 'randn', 'connectivity', 1.0, 'radius', 1.0));
% Connect every reservoir to  output
net.setConnection(lID(4), lID(3), struct('initType', 'randn', 'connectivity', 1.0, 'radius', 1.0));
% net.setConnection(lID(5), lID(3), struct('initType', 'randn', 'connectivity', 1.0, 'radius', 1.0));
% net.setConnection(lID(6), lID(3), struct('initType', 'randn', 'connectivity', 1.0, 'radius', 1.0));
% Connect input to output
net.setConnection(lID(2), lID(3), struct('initType', 'randn', 'connectivity', 1.0, 'radius', 1.0));
% Connect reservoirs in a feed-forward structure 
net.setConnection(lID(2), lID(4), struct('initType', 'randn', 'connectivity', 1.0, 'radius', 1.0));
% net.setConnection(lID(4), lID(5), struct('initType', 'randn', 'connectivity', 0.01, 'radius', 1.0));
% net.setConnection(lID(5), lID(6), struct('initType', 'randn', 'connectivity', 0.01, 'radius', 1.0));

%net.setConnection(lID(4), lID(3), struct('initType', 'randn', 'connectivity', 1.0, 'radius', 1.0));

% connections = [1.0, 1.0, 1.0, 1.0, 1.0, 0.1];
% radii = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
% initType = {'randn', 'randn', 'randn', 'randn', 'randn', 'randn'};
% arch = sparse([lID(2) lID(2) lID(1) lID(1) lID(4) lID(4)],...
%               [lID(4) lID(3) lID(3) lID(4) lID(3) lID(4)], connections, 6, 6);
          
%net.setConnections(arch, initType, radii);

% net.visualize();

%% Train the network
boltrain = BatchOutputLayerTrain();
%net.setConnectivity(0.01, lID(4));
initLen = 1;
tic
x = boltrain.train(net, train_input, train_output, initLen, 'ridge2'); %0.18 %2.70 %19996 
toc
% figure, plot(x')

%% Generate
seq_length = 25;
vocab_size = length(charSet);
tic
net.rememberStates();
%y = net.generate(train_input(:,1), 1000);
y = net.predict(train_input(:, 1:seq_length));
net.recallStates();
toc

for k = 1:size(y, 2)
    [val, arg] = max(y(:,k));
    generatedText(k) = charSet(arg);
end
char(generatedText)
figure, plot(y');
%generatedText = char(y * input_std + input_mean);

smooth_loss = -log(1.0 / vocab_size) * seq_length;
net.rememberStates();
for pos = 1:seq_length:floor(length(train_output)/seq_length)*seq_length
    
    %y = net.generate(train_input(:,1), 1000);
    y = net.predict(train_input(:, pos:pos+seq_length-1));
    
    sm_y = exp(y)./sum(exp(y));
    loss = 0;
    for l = 0:seq_length-1
        loss  = loss - log(sm_y(train_output(:, pos + l) == 1, l+1));
    end
    loss
    smooth_loss = smooth_loss * 0.999 + loss * 0.001;
end
net.recallStates();
smooth_loss



loss = -sum(truePDF .* log(sm_y))


ps{t,1} = exp(ys{t,1}) / sum(exp(ys{t,1}));
loss  = loss - log(ps{t,1} (targets(t,1),1) );
smooth_loss = smooth_loss * 0.999 + loss * 0.001;


