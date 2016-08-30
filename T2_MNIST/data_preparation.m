%% Load MNIST dataset
mnisttrain = importdata('./mnistCSV/mnist_train.csv',',');
mnisttest = importdata('./mnistCSV/mnist_test.csv', ',');
train_input = mnisttrain(:,2:end)';
train_output = mnisttrain(:,1)';
test_input = mnisttest(:,2:end)';
test_output = mnisttest(:,1)';


img = reshape(test_input_ext(:,18), [28, 28])';
figure, hold on;
subplot(2,1,1), imshow(uint8(img));
trImgs = uint8(generateProjections(img, 15, 0, 0));
subplot(2,1,2), imshow(round(trImgs));
hold off;
%% Take an image and deform it using projective transformation
theta = [0];    % rotation around y axis
phi = [-5:5:5];     % rotation around x axis
ksi = [-15:15:15];              % rotation around z axis
[test_input_ext, test_output_ext] = extImageSetByProjTr(test_input, test_output, theta, phi, ksi);
[train_input_ext, train_output_ext] = extImageSetByProjTr(train_input, train_output, theta, phi, ksi);

mnisttest_ext = [test_output_ext; test_input_ext]';
csvwrite('./mnist_test_ext.csv', mnisttest_ext);

mnisttrain_ext = [train_output_ext; train_input_ext]';
csvwrite('./mnist_train_ext.csv', mnisttrain_ext);

