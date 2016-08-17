%% Load MNIST dataset
mnisttrain = importdata('./mnist_train.csv',',');
mnisttest = importdata('./mnist_test.csv', ',');
train_input = mnisttrain(:,2:end)';
train_output = mnisttrain(:,1)';
test_input = mnisttest(:,2:end)';
test_output = mnisttest(:,1)';
trainSize = size(train_input,2);

img = reshape(train_input(:,18), [28, 28])';
figure, hold on;
subplot(2,1,1), imshow(uint8(img));
trImgs = uint8(generateProjections(img, 0, -10, -10));
subplot(2,1,2), imshow(round(trImgs));
hold off;
%% Take an image and deform it using projective transformation
theta = [-15:15:15];        % rotation around y axis
phi = [-10:10:10];     % rotation around x axis
ksi = [0]; % rotation around z axis
trNum = length(theta) * length(phi) * length(ksi);
train_input_ext = zeros(size(train_input,1), size(train_input,2) * trNum);
train_output_ext = zeros(size(train_output,1), size(train_output,2) * trNum);
counter = 1;
for j = 1:size(train_input,2)
    j/trainSize
    img = reshape(train_input(:,j), [28, 28])';
    trImgs = uint8(generateProjections(img, theta, phi, ksi));
    for k = 1:size(trImgs,3)
        img = trImgs(:,:,k);
        img = img(:);
        img(isnan(img)) = 0;
        train_input_ext(:, counter) = img;
        train_output_ext(:, counter) = train_output(:,j);
        counter = counter + 1;
    end
end

mnisttrain_ext = [train_output_ext; train_input_ext]';
csvwrite('./mnist_train_ext.csv', mnisttrain_ext);

