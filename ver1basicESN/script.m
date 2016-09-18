%% Problem 1: Mackeyglass system
% http://www.macalester.edu/~kaplan/knoxville/science1977.pdf
% (a) Run the code in the following section, you might have to run it a few 
% times to produce a good predction. 
% (b) Select the predicted/generted process with the minimum  MSE.
% (c) Explain why ESN in the predictive mode produces much smaller error.
figure
mackeyglass_data = mackeyglass(5000);
mid_point = round(length(mackeyglass_data) / 2);
train_input = mackeyglass_data(1:mid_point, :);
train_output = mackeyglass_data(2:mid_point + 1, :);
test_input = mackeyglass_data(mid_point + 1:end - 1, :);
test_output = mackeyglass_data(mid_point + 2:end, :);


N = 20;
connectivity = 0.5;
sp_radius = 1.25;
lr = 0.5;
[Win, W] = buildESN(size(train_input,2), N, connectivity, sp_radius);
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
plot( test_output(1:200));
%plot( Yg(1:200));
plot( Yp(1:200));
hold off;
axis tight;
legend('Target signal', 'Generated');

figure(2), hold;
plot(states_evolution(:, 1:500)')
title('States evolution');


%% Problem 2: Lorenz system
% https://www.math.auckland.ac.nz/~hinke/preprints/okh_icdea_preprint.pdf
% (a) Run the code in the following section;
% (b) Select the parameters of connectivity, speactral radius (sp_radius), and 
% leakage rate(lr) that generate a trajactory that follows the trajectory
% of the Lorenz attractor. Try to keep N lower, however the accuracy is the
% most important criterion; For every set of parameters run the ESN 10
% times, calculate the mean and standard deviation. Print out the set of 
% parameters that produces minimal MSE, and visualize the trajectory.
% (c) Explain based on your experience how each of the parameters effect
% the accuracy.
[x y z] = lorenz(28, 10, 8/3);
Lorenz_data = [x, y, z];

mid_point = round(length(Lorenz_data) / 2);
train_input = Lorenz_data(1:mid_point, :);
train_output = Lorenz_data(2:mid_point + 1, :);
test_input = Lorenz_data(mid_point + 1:end - 1, :);
test_output = Lorenz_data(mid_point + 2:end, :);

N = 10;
connectivity = 0.25;
sp_radius = 1;
lr = 0.05;
[Win, W] = buildESN(size(Lorenz_data,2), N, connectivity, sp_radius);
[Wout, states, states_evolution] = trainESN(train_input, train_output, Win, W, lr, 0);
Y = runESN(test_input(1,:), length(test_output), states, Win, W, Wout, lr, 0); 
figure, plot(states_evolution')

errorLen = 1000;
dif = Y(1:errorLen, :) - test_output(1:errorLen, :);
mse = sqrt(sum(diag(dif' * dif)));
disp( ['MSE = ', num2str( mse, '%5.2f' )] );

figure(3), hold on;
plot3(test_input(:,1), test_input(:,2), test_input(:,3));
plot3(Y(:,1), Y(:,2), Y(:,3));

figure(4);hold on;
plot( test_output(1:4000,:), 'k');
plot( Y(1:4000,:));
hold off;
axis tight;
legend('Target signal','Target signal', 'Target signal',  'Generated',  'Generated',  'Generated');

%% Problem 3: Incorporate to the learning process the following parameters by 
% modifying predictESN and runESN functions.
% (a) noiseLevel, add to the state update step
% (b) add feedback connection from the output to the input, regulate the 
% contribution of this connection by teacherScaling parameter.
% (c) Experiment with the Lorenz system by changing the noiseLevel and 
% teacherScaling parameters.
% Comment on the contribution of these parameters to generative modeling.

% Refernce:
% (a) http://minds.jacobs-university.de/sites/default/files/uploads/papers/PracticalESN.pdf



%% Problem 4: Sinewave generator
% Use the modified above ESN with the feedback teacher connections and
% added noise parameter to learn sinewave generator.
% The input signal u(n) is a slowly varying frequency setting, the desired 
% output y(n) is a sinewave of a frequency indicated by the current input. 
% Assume that a training input-output sequence D=(u(1),y(1)),...,(u(nmax),
% y(nmax)) is given (see the input and output signals in ; here the input 
% is a slow random step function indicating frequencies ranging from 1/16 
% to 1/4 Hz). The task is to train a ESN from these training data such that 
% on slow test input signals, the output is again a sinewave of the 
% input-determined frequency.
% Reference: http://www.scholarpedia.org/article/Echo_state_network 
sequenceLength = 5000;
outMinPeriod = 4 ; outMaxPeriod = 16; % output period length range
superPeriod = 200; % input period length
[inputSequence, outputSequence] = generate_freqGen_sequence(sequenceLength,... 
                                  outMinPeriod, outMaxPeriod, superPeriod); 
inputSequence = inputSequence(:,2);

train_fraction = 0.8 ; % use train_fraction of data in training, rest in testing
[trainInputSequence, testInputSequence] = split_train_test(inputSequence, train_fraction);
[trainOutputSequence,testOutputSequence] = split_train_test(outputSequence, train_fraction);

N = 500;
connectivity = 0.1;
sp_radius = 0.5;
lr = 1;
% Specify teacherScaling and pass it to your implementation of trainESN
% Specify noiseLevel and pass it to you implementation of trainESN
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

%% Problem 5: the Wiener-Hopf solution
% See slide 7, for details on the Wiener-Hopf method.
% Modify trainESN function, such that depending on the input paramter 'method'
% either pseudoinverse or Wiener-Hopf method is used to find weights of
% the output layer. Right now trainESN uses only pseudoinverse to find ouput
% weights.