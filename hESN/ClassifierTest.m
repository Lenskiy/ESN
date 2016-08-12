%% Load MNIST Training & Test data

[train, test] = MNIST('/Users/dakrefl/data/',1000,2);

train_in  = train(:,2:size(train,2));
train_tar = train(:,1);

test_in   = test(:,2:size(test,2));
test_tar  = test(:,1);



%% ESN TEST

nNodes = 500;

architecture = struct('inputDim',   1, ...
                      'numNodes',   nNodes, ... 
                      'outputDim',  10);      
parameters  = struct('node_type',      'tanh',... 
                     'radius',      1, ...
                     'leakage',     1, ... 
                     'connectivity',0.1,...
                     'init_type', 'rand');  
                 
esn = ESN(architecture, parameters);

train = RRClassifierTrain();

tic()

train.train(esn, train_in, train_tar, 10); 

toc()


