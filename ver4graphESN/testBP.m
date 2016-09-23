net = Network();

lID(1) = net.addLayer(1, 'bias',   struct('nodeType', 'linear', 'leakage', 1, 'initType', 'randn')); % All layers have bias
lID(2) = net.addLayer(size(train_input,1), 'input',  struct('nodeType', 'linear', 'leakage', 1.0));
lID(3) = net.addLayer(size(train_output,1), 'output', struct('nodeType', 'linear', 'leakage', 1.0));
lID(4) = net.addLayer(200, 'reservoir', struct('nodeType', 'tanh', 'radius', 1.0, 'leakage', 0.2, 'connectivity',0.25, 'initType', 'randn'));
lID(5) = net.addLayer(200, 'reservoir', struct('nodeType', 'tanh', 'radius', 0.6, 'leakage', 0.6, 'connectivity',0.15, 'initType', 'randn'));
lID(6) = net.addLayer(50, 'layer', struct('nodeType', 'tanh', 'radius', 0.5, 'leakage', 1.0, 'connectivity',0.1, 'initType', 'randn'));
lID(7) = net.addLayer(50, 'layer', struct('nodeType', 'tanh', 'radius', 0.5, 'leakage', 1.0, 'connectivity',0.1, 'initType', 'randn'));
lID(8) = net.addLayer(100, 'reservoir', struct('nodeType', 'tanh', 'radius', 0.6, 'leakage', 0.6, 'connectivity',0.15, 'initType', 'randn'));


connections = [1.0, 1.0, 1.0, 1.0  1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0];
arch = sparse([lID(1) lID(1) lID(1) lID(1) lID(1) lID(1) lID(2) lID(2) lID(4) lID(5) lID(6) lID(7) lID(8)],...
              [lID(3) lID(4) lID(5) lID(6) lID(7) lID(8) lID(4) lID(5) lID(6) lID(7) lID(8) lID(8) lID(3)], connections,8,8);
net.setConnections(arch, 'randn');

net.visualize();