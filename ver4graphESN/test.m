net = Network();
% All networks have bias
lID(1) = net.addLayer(1, 'bias',   struct('nodeType', 'linear', 'leakage', 1, 'initType', 'randn'));
lID(2) = net.addLayer(1, 'input',  struct('nodeType', 'linear', 'leakage', 1.0));
lID(3) = net.addLayer(1, 'output', struct('nodeType', 'linear', 'leakage', 1.0));
lID(4) = net.addLayer(1000, 'reservoir', struct('nodeType', 'tanh', 'radius', 0.9, 'leakage', 0.9, 'connectivity',0.1, 'initType', 'randn'));
% (1) add a bais type of layer maybe make it default one
% (2) add noise paramter to layers' params
% (3) add withend reservoir 
% (4) rewrite sprand and sprandn
% (5) add/reset states

connections = [1.0, 1.0, 1.0, 1.0, 1.0];
arch = sparse([lID(2) lID(2) lID(4) lID(1) lID(1)],...
              [lID(4) lID(3) lID(3) lID(3) lID(4)], connections,6,6);
net.setConnections(arch, 'randn');
%net.W
%net.visualize();


boltrain = BatchOutputLayerTrain();
initLen = 100;
[~, x] = boltrain.train(net, train_input, train_output, initLen);

% figure, plot(x')
tic
y = net.generate(test_input);
toc
figure, hold on;
plot(y);
plot(test_output);