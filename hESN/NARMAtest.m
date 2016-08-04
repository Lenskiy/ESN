%% NARMA training/testing data INIT

data = NARMA10series(1600);

testp = 0.5;

split_point = round((1-testp) * length(data)); % take testp as hold out test set

train_input  = data(1, 1:split_point,1);
train_output = data(2, 2:split_point + 1);
test_input   = data(1, split_point + 1:end - 1);
test_output  = data(2, split_point + 2:end);



%% ESN TEST

nNodes = 100;

architecture = struct('inputDim',   size(train_input,1), ...
                      'numNodes',   nNodes, ... 
                      'outputDim',  size(train_output,1));      
parameters  = struct('node_type',      'tanh',... 
                     'radius',      0.8, ...
                     'leakage',     0.2, ... 
                     'connectivity',0.1,...
                     'init_type', 'rand');  
esn = ESN(architecture, parameters);

train = LTrain();

initL = 100;

tic()
train.train(esn, train_input, train_output, initL); 
%Y = esn.generate(test_input(1, :), size(test_output(1,:),2), 1);
Y = esn.predict(test_input(1, :), 1);
toc()
NRMSE(Y,test_output(1,:))


figure, hold on; title('NARMA10 test');
plot(test_output(1,:));
plot(Y(1,:));


%% SESN TEST

sArchitecture = struct('inputDim',  size(train_input,1), ...
                      'numNodes',   [100; 100], ... 
                      'outputDim',  size(train_output,1));
                   
sParameters(1)  = struct('node_type',      'tanh',... 
                     'radius',      0.8, ...
                     'leakage',     0.2, ... 
                     'connectivity',0.1,...
                     'init_type', 'rand');  
sParameters(2)  = struct('node_type',      'tanh',... 
                     'radius',      0.2, ...
                     'leakage',     0.5, ... 
                     'connectivity',0.05,...
                     'init_type', 'rand'); 
                 
sESN = StackedESN(sArchitecture, sParameters, 'rand');
train = RRTrain();
train.train(sESN, train_input, train_output, 100);
%Y = esn.generate(test_input(1, :), size(test_output(1,:),2), 1);
Y = esn.predict(test_input(1, :), 1);

NRMSE(Y,test_output(1,:))


figure, hold on; title('NARMA10 test');
plot(test_output(1,:));
plot(Y(1,:));


