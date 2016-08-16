%% Load MNIST dataset
mnisttrain = importdata('./mnist_train.csv',',');
mnisttest = importdata('./mnist_test.csv', ',');
train_input = mnisttrain(:,2:end)';
train_output = mnisttrain(:,1)';
test_input = mnisttest(:,2:end)';
test_output = mnisttest(:,1)';
trainSize = size(train_input,2);

%% Take an image and deform it using projective transformation
theta = [-5:5:5];
phi = [-5:5:5];
ksi = [-10:10:10];
trNum = length(theta) * length(phi) * length(ksi);
train_input_ext = zeros(size(train_input,1) * trNum, size(train_input,2));
train_output_ext = zeros(size(train_output,1) * trNum, size(train_output,2));
counter = 1;
for j = 1:size(train_input,1)
    j/trainSize
    img = reshape(train_input(14,:), [28, 28])';
    trImgs = generateProjections(img, theta, phi, ksi);
    img = resI(:,:,k);
    
    for k = 1:size(resI,3)
        img = resI(:,:,k);
        img = img(:);
        img(isnan(img)) = 0;
        train_input_ext(counter, :) = img;
        train_output_ext(counter, :) = train_output(j,:);
        counter = counter + 1;
    end
end


