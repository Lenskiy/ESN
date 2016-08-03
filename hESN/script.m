%% Mackeyglass training/testing data
mackeyglass_data = mackeyglass(2100);
mackeyglass_data = mackeyglass_data(101:2100);
mid_point = round(15 * length(mackeyglass_data) / 20); % take a small portion for training
train_input = mackeyglass_data(1:mid_point, :);
train_output = mackeyglass_data(2:mid_point + 1, :);
test_input = mackeyglass_data(mid_point + 1:end - 1, :);
test_output = mackeyglass_data(mid_point + 2:end, :);

esn = ESN('tanh', 1000, 1, 0.9, 0.6, 0.1, 'randn');
train = Train();
train.train(esn, train_input, train_output, 100) % do not change the state, or be able to restart the state to the begining
Y = esn.generate(test_input(1, :), size(test_output(:,1),1), 1);
%Y = esn.generate(train_input(1, :), size(train_output(:,1),1), 1);

figure, hold on; title('Mackeyglass system');
plot(test_input(:,1));
plot(Y(:,1));