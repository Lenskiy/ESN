%% Lorenz training/testing data
% [x y z] = lorenz(28, 10, 8/3);
% Lorenz_data = [x, y, z];

% mid_point = round((length(Lorenz_data) - 1) / 2);
% train_input = Lorenz_data(1:mid_point, :);
% train_output = Lorenz_data(2:mid_point + 1, :);
% test_input = Lorenz_data(mid_point + 1:end - 1, :);
% test_output = Lorenz_data(mid_point + 2:end, :);

%% NARMA10 training/testing data
% NARMA10_data = NARMA10(5000);
% mid_point = round(19 * length(NARMA10_data) / 20); % take a small portion for training
% train_input = NARMA10_data(1:mid_point, :);
% train_output = NARMA10_data(2:mid_point + 1, :);
% test_input = NARMA10_data(mid_point + 1:end - 1, :);
% test_output = NARMA10_data(mid_point + 2:end, :);

%% Mackeyglass training/testing data
mackeyglass_data = mackeyglass(2000);
mid_point = round(15 * length(mackeyglass_data) / 20); % take a small portion for training
train_input = mackeyglass_data(1:mid_point, :);
train_output = mackeyglass_data(2:mid_point + 1, :);
test_input = mackeyglass_data(mid_point + 1:end - 1, :);
test_output = mackeyglass_data(mid_point + 2:end, :);

% N = [10, 10, 10, 10];
% lr = [0.9 0.1, 0.5, 0.25];
% sp_radius = [0.1 0.9, 0.5, 0.25];
% connectivity = [0.1 0.1 0.1, 0.1];
numESNs = 10;
N = round(10 * rand(numESNs,1) + 10);
lr = rand(numESNs,1);
sp_radius = rand(numESNs,1);
connectivity = 0.2 * rand(numESNs,1) + 0.01;
sigma_noise = 0.00;

[Win, W] = buildStackedESN(size(train_input,2), N, connectivity, sp_radius, 'randn');
[Wout, states, states_evolution, tf] = trainStackedESN(train_input, train_output, Win, W, lr, sigma_noise, 'tanh', 'linear');
Y = genStackedESN(test_input(1,:), length(test_output), states, Win, W, Wout, lr, sigma_noise, 'tanh', 'pca', 1, tf);

figure, 
subplot(2,1,1), hold on
axis([1,size(states_evolution,2), -1, 1])
d1 = plot(1:size(states_evolution,2), states_evolution(1:N(1),:)','r');
d2 = plot(1:size(states_evolution,2), states_evolution(N(1)+1:end,:)','g');
d3 = plot(1:size(states_evolution,2), states_evolution(N(1)+N(2)+1:end,:)','b');
d4 = plot(1:size(states_evolution,2), states_evolution(N(1)+N(2)+N(3)+1:end,:)','k');
title('States during the learning stage')
legend([d1(1), d2(1), d3(1), d4(1)],['\rho_1 = ', num2str(sp_radius(1)), '  l_1 = ', num2str(lr(1))],...
                      ['\rho_2 = ', num2str(sp_radius(2)), '  l_2 = ', num2str(lr(2))],...
                      ['\rho_3 = ', num2str(sp_radius(3)), '  l_3 = ', num2str(lr(3))],...
                      ['\rho_4 = ', num2str(sp_radius(4)), '  l_4 = ', num2str(lr(4))]);

subplot(2,1,2), hold on
title('Mackeyglass system');
plot(test_output(:,1));
plot(Y(:,1));

% figure(3), hold on;
% title('Lorenz system');
% plot3(test_output(:,1), test_output(:,2), test_output(:,3));
% plot3(Y(:,1), Y(:,2), Y(:,3));


%% Find a best combination of parameters for the ESNs 
% define the parameters of the tested ESNs    
ESNtype = {'randn', 'tanh', 'linear'};
nTrials = 100;
nESNs = 2;

            %numNodes, leakRate, spectalRadii, connectivity
sParams{1} = {[100], [0.1:0.1:1], [0.1:0.1:1.1], [0.05 0.1]};  % ESN1  
sParams{2} = {[100], [0.1:0.1:1], [0.1:0.1:1.1], [0.05 0.1]};  % ESN2

sParams{1} = {[20], [.2:0.2:0.8], [0.8], [0.1]};  % ESN1  
sParams{2} = {[20], [.2:0.2:0.8], [0.8], [0.1]};  % ESN2

% Evaluate all combinations of paramters 
[mse_results, mse_results_std, parameters_grid, best_mse, best_paramters] = ...
    gridSearchESNparamters(train_input, train_output, test_input(1,:),...
    test_output, nTrials, nESNs, ESNtype, sParams);

% put inf for the parameters of interest that will be selected;
% other parameters are fixed
%selecting_params = [20; 0.3; inf; 0.1; 20; 1.0; inf; 0.1];
sParams{1} = {100, inf, 0.2, 0.05};  % ESN1  
sParams{2} = {100, inf, 0.2, 0.05};  % ESN2

% select the parameters 
[errMap, X, Y] = getErrorMap(parameters_grid, mse_results, sParams);
 
figure;
surf(X, Y, errMap);






          %numNodes, leakRate, spectalRadii, connectivity
sParams{1} = {100, inf, 1.1, 0.1};  % ESN1  
sParams{2} = {100, inf, 1.1, 0.1};  % ESN2

% select the parameters 
[errMap, X, Y] = getErrorMap(parameters_grid, mse_results, sParams);
 
figure;
surf(X, Y, errMap);


sParams{1} = {100, 1.0, inf, 0.1};  % ESN1  
sParams{2} = {100, 1.0, inf, 0.1};  % ESN2

% select the parameters 
[errMap, X, Y] = getErrorMap(parameters_grid, mse_results, sParams);
 
figure;
surf(X, Y, errMap);


%title(['MSE (connectivity = ',num2str(connectivity(connect_ind)),')']);
% ax = gca;
% ax.XTick = [1:length(sp_radius1)];
% ax.YTick = [1:length(sp_radius2)];
% ax.XTickLabel = sp_radius1;
% ax.YTickLabel = sp_radius2;
% xlabel(['\rho_1(l = ',num2str(lr(1)),')']);
% ylabel(['\rho_2(l = ',num2str(lr(2)),')']);







%% Find a best combination of spectral radii for ESN1 and ESN2 and the degree of connectivity
% N_trials = 100;
% lr = [0.2 0.9];
% sigma_noise = 0.0;
% N = [20 20];
% connectivity=0.05:0.05:0.5;
% sp_radius1=0.05:0.1:1.05;
% sp_radius2=0.05:0.1:1.05;
% errorLen = length(test_input);
% err_tensor_m = zeros(length(sp_radius1), length(sp_radius2), length(connectivity));
% err_tensor_s = zeros(length(sp_radius1), length(sp_radius2), length(connectivity));
% err_tensor_cor = zeros(length(sp_radius1), length(sp_radius2), length(connectivity));
% counter = 1;
% total_inters = length(sp_radius1) * length(sp_radius2) * length(connectivity);
% disp('Starting...');
% for c = 1:length(connectivity)
%     for r2 = 1:length(sp_radius2);
%         for r1 = 1:length(sp_radius1);
%             counter = counter + 1;
%             clear mse;
%             for i = 1:N_trials
%                 %i
%                 [Win, W] = buildStackedESN(size(train_input,2), N, [connectivity(c) connectivity(c)], [sp_radius1(r1); sp_radius2(r2)], 'randn');
%                 [Wout, states, states_evolution] = trainStackedESN(train_input, train_output, Win, W, lr, sigma_noise, 'tanh', 'linear');
%                 Y = runStackedESN(test_input(1,:), length(test_output), states, Win, W, Wout, lr, sigma_noise, 'tanh', 'linear', 1); 
%                 dif = Y(1:errorLen, :) - test_output(1:errorLen, :);
%                 mse(i) = sqrt(trace(dif' * dif));
%             end
%             
%             err_tensor_m(r1, r2, c) = mean(mse(find(mse < 50)));
%             err_tensor_s(r1, r2, c) = std(mse(find(mse < 50)));
%             ccfunc = xcorr(test_output - mean(test_output), Y - mean(Y), 'coeff');
%             err_tensor_cor(r1, r2, c) = max(ccfunc);
%             disp(   ['progress: ' num2str(100*counter/total_inters,'%2.2f'),'%',...
%                 ', c = ',num2str( connectivity(c), '%2.2f'),...
%                 ', rho_1 = ', num2str( sp_radius1(r1), '%2.2f'),...
%                 ', rho_2 = ', num2str( sp_radius2(r2), '%2.2f' ),...
%                 ', m = ',num2str( err_tensor_m(r1, r2, c) ),...
%                 ', s = ',num2str(err_tensor_s(r1, r2, c)),...
%                 ', c = ',num2str( err_tensor_cor(r1, r2, c) )]);
%         end
%     end
% end
% 
% connect_ind = 10;  
% 
% figure;
% surf(err_tensor_cor(:,:,connect_ind));
% title(['Correlation coefficient (connectivity = ',num2str(connectivity(connect_ind)),')']);
% ax = gca;
% ax.XTick = [1:length(sp_radius1)];
% ax.YTick = [1:length(sp_radius2)];
% ax.XTickLabel = sp_radius1;
% ax.YTickLabel = sp_radius2;
% xlabel(['\rho_1(l = ',num2str(lr(1)),')']);
% ylabel(['\rho_2(l = ',num2str(lr(2)),')']);
% 
% figure;
% surf(err_tensor_m(:,:,connect_ind));
% title(['MSE (connectivity = ',num2str(connectivity(connect_ind)),')']);
% ax = gca;
% ax.XTick = [1:length(sp_radius1)];
% ax.YTick = [1:length(sp_radius2)];
% ax.XTickLabel = sp_radius1;
% ax.YTickLabel = sp_radius2;
% xlabel(['\rho_1(l = ',num2str(lr(1)),')']);
% ylabel(['\rho_2(l = ',num2str(lr(2)),')']);

