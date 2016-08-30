load('/Users/artemlenskiy/Desktop/cifar-10-batches-mat/test_batch.mat');
load('/Users/artemlenskiy/Desktop/cifar-10-batches-mat/data_batch_1.mat');
load('/Users/artemlenskiy/Desktop/cifar-10-batches-mat/data_batch_2.mat');
load('/Users/artemlenskiy/Desktop/cifar-10-batches-mat/data_batch_3.mat');
load('/Users/artemlenskiy/Desktop/cifar-10-batches-mat/data_batch_4.mat');
load('/Users/artemlenskiy/Desktop/cifar-10-batches-mat/data_batch_5.mat');


numSamples = size(data,1);
dim = size(data,2)/3;
inputData = zeros(numSamples , dim + 1);
inputData(:, 1) = labels;
for k = 1:size(data, 1)
    im = reshape(data(k,:), [32, 32, 3]);
    imGrey = rgb2gray(im);
    inputData(k, 2:end) = imGrey(:);
end

%csvwrite('./cifar_test.csv', inputData);
csvwrite('./cifar_train_batch3.csv', inputData);

% test
figure, imshow(uint8(reshape(inputData(5, 2:end), 32, 32)));