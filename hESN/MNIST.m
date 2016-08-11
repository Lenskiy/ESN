function [ Train, Test ] = MNIST( path, Ntrain, Ntest )
    % Reads the MNIST dataset from .csv
    % First pos is target, rest is pic line-wise scanned (28x28=784)
    % Each row corresponds to one image
    M_train = csvread(strcat(path,'mnist_train.csv'));
    M_test  = csvread(strcat(path,'mnist_test.csv'));

    Train = M_train(1:Ntrain,:);
    Test  = M_test(1:Ntest,:);

end

