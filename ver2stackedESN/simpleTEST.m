
%% Simple test script
%data = mackeyglass(1600);

data = meanAdjSdevNorm(mackeyglass(1600));

testp = 0.5;

MCruns = 10;

split_point = round((1-testp) * length(data)); % take testp as hold out test set

train_input  = data(1:split_point, :);
train_output = data(2:split_point + 1, :);
test_input   = data(split_point + 1:end - 1, :);
test_output  = data(split_point + 2:end, :);

% ESN settings
N = [20, 20, 20];
connectivity = [0.1, 0.1, 0.1];
sp_radius = [0.1, 0.5, 1];
lr = [0.1,0.5, 0.9];
sigma_noise = 0.1;

tic

% Monte Carlo runs
M = [];
for r = 1 : MCruns 


    [Win, W] = buildStackedESN(size(train_input,2), N, connectivity, sp_radius, 'randn');
    [Wout, states, states_evolution] = trainStackedESN(train_input, train_output, Win, W, lr, sigma_noise, 'tanh', 'linear');
    Y = runStackedESN(test_input(1,:), length(test_output), states, Win, W, Wout, lr, sigma_noise, 'tanh', 'linear', 1);

    error = RMSE(Y(:,1),test_output(:,1));
    
    if error < 1
        M = [M Y];
    end
end

% Average over results
Y = mean(M,2);

% Plot
figure('name','ESN test'), 
subplot(2,1,1), hold on
axis([1,size(states_evolution,2), -1, 1])
d1 = plot(1:size(states_evolution,2), states_evolution(1:N(1),:)','r');
d2 = plot(1:size(states_evolution,2), states_evolution(N(1)+1:end,:)','k');

error = RMSE(Y(:,1),test_output(:,1));
sprintf('RMSE: %d', error)

title('States during the learning stage')
legend([d1(1), d2(1)],['\rho_1 = ', num2str(sp_radius(1)), '  l_1 = ', num2str(lr(1))],...
                      ['\rho_2 = ', num2str(sp_radius(2)), '  l_2 = ', num2str(lr(2))]);

subplot(2,1,2), hold on
title('ESN test');
plot(test_output(:,1));
plot(Y(:,1));
toc
