%% Problem 1: Mackeyglass system
% (a) Run the code in the following section, you might have to run it a few 
% times to produce a good predction. 
% (b) Select the best in terms of the minimum of the MSE values.
% (c) Explain why ESN in predictive mode produces much smaller error.
figure
mackeyglass_data = mackeyglass(5000);
mid_point = round(length(mackeyglass_data) / 2);
train_input = mackeyglass_data(1:mid_point, :);
train_output = mackeyglass_data(2:mid_point + 1, :);
test_input = mackeyglass_data(mid_point + 1:end - 1, :);
test_output = mackeyglass_data(mid_point + 2:end, :);


N = 20;
connectivity = 0.2;
sp_radius = 1.25;
lr = 0.5;
[Win, W] = buildESN(size(data,2), N, connectivity, sp_radius);
[Wout, states, states_evolution] = trainESN(train_input, train_output, Win, W, lr);
% generative mode
Yg = runESN(test_input(1, :), length(test_input), states, Win, W, Wout, lr); 
% predictive mode
Yp = predictESN(test_input, states, Win, W, Wout, lr); 


errorLen = 50;
mse = norm(Yg(1:errorLen) - test_output(1:errorLen));
disp( ['Generative mode MSE = ', num2str( mse )] );
mse = norm(Yp(1:errorLen) - test_output(1:errorLen));
disp( ['Predictive mode MSE = ', num2str( mse )] );

% plot some signals
figure(1);hold on;
plot( test_output);
plot( Yg);
plot( Yp);
hold off;
axis tight;
legend('Target signal', 'Generated', 'Predicted');

figure(2), hold;
plot(states_evolution')
title('States evolution');


%% Problem 2: Lorenz system
% (a) Run the code in the following section;
% (b) Select the parameters of connectivity, speactral radius (sp_radius), and 
% leakage rate that generated trajactory following the buterfly trajectory
% of the Lorenz attractor. Try to keep N lower, however the accuracy is the
% most important criterion; For every set of parameters run the ESN 10
% times, calculate the mean and standard deviation. Print outthe set of parameters
% that produces minimal MSE, and visualize the trajectory.
% (c) Explain based on your experience how each of the parameters effect
% the accuracy.
[x y z] = lorenz(28, 10, 8/3);
Lorenz_data = [x, y, z];

mid_point = round(length(Lorenz_data) / 2);
train_input = Lorenz_data(1:mid_point, :);
train_output = Lorenz_data(2:mid_point + 1, :);
test_input = Lorenz_data(mid_point + 1:end - 1, :);
test_output = Lorenz_data(mid_point + 2:end, :);

N = 50;
connectivity = 0.2;
sp_radius = 1.25;
lr = 0.3;
[Win, W] = buildESN(size(Lorenz_data,2), N, connectivity, sp_radius);
[Wout, states] = trainESN(train_input, train_output, Win, W, lr);
% either predictive or generative modes are supported
Y = runESN(test_input(1,:), length(test_output), states, Win, W, Wout, lr); 


errorLen = 50;
dif = Y(1:errorLen, :) - test_output(1:errorLen, :);
mse = sum(sqrt(diag(dif' * dif)));
disp( ['MSE = ', num2str( mse )] );

figure(3), hold on;
plot3(test_input(:,1), test_input(:,2), test_input(:,3));
plot3(Y(:,1), Y(:,2), Y(:,3));

% figure(4);hold on;
% plot( test_output, 'k');
% plot( Y);
% hold off;
% axis tight;
% legend('Target signal', 'Generated');

%% Problem 3

sequenceLength = 5000;
outMinPeriod = 4 ; outMaxPeriod = 16; % output period length range
superPeriod = 200; % input period length
[inputSequence, outputSequence] = generate_freqGen_sequence(sequenceLength, outMinPeriod, outMaxPeriod, superPeriod) ; 
inputSequence = inputSequence(:,2);

train_fraction = 0.8 ; % use train_fraction of data in training, rest in testing
[trainInputSequence, testInputSequence] = split_train_test(inputSequence, train_fraction);
[trainOutputSequence,testOutputSequence] = split_train_test(outputSequence, train_fraction);

N = 500;
connectivity = 0.1;
sp_radius = 0.5;
lr = 1;
[Win, W] = buildESN(size(trainInputSequence,2), N, connectivity, sp_radius);
[Wout, states] = trainESN(trainInputSequence, trainOutputSequence, Win, W, lr);
Y = predictESN(testInputSequence, states, Win, W, Wout, lr); 

close
figure(4);hold on;
plot(testInputSequence)
%plot( testOutputSequence, 'k');
plot( Y);
hold off;
axis tight;
legend('Target signal', 'Generated');
