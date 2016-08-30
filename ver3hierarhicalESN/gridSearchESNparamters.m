
function [mse_results, mse_results_std, parameters_grid, best_mse, best_paramters] = gridSearchESNparamters(training_input, training_output, testing_input, testing_output,...
                                                                    nTrials, ESNtype, sParams)

    feedback_scaling = 1;
    stdNoise = 0;
    input_weight_type = ESNtype{1};
    node_type = ESNtype{2};
    output_type = ESNtype{3};
    numESNparams = size(sParams{1},2);
    
    input_size = size(testing_input,2);
    test_data_size = size(testing_output,1);
    allESNparameters = [];
    for k = 1:length(sParams)
        allESNparameters = [allESNparameters sParams{k}];
    end
    parameters_grid = cartprod(allESNparameters);
    numParamSets = length(parameters_grid)

    
    mse_results = inf .* ones(1,numParamSets);
    mse_results_std = inf .* ones(1,numParamSets);
    parfor experiment = 1:numParamSets
        %disp(['progress: ', num2str(experiment/numParamSets, '%2.2f')]);
        % Split parameters to corresponding variables
        p = parameters_grid(experiment,:);
        numNodes = p(1:numESNparams:end);
        connectivity = p(2:numESNparams:end);
        spRadius = p(3:numESNparams:end);
        leakRate = p(4:numESNparams:end);
        mse_mean = 0;
        mse_std = 0;
        successful_run_counter = 0;
        for k = 1:nTrials
            [sWin, sW] = buildStackedESN(input_size, numNodes,...
                                        connectivity,...
                                        spRadius,...
                                        input_weight_type);
            [sWout, x, ~] = trainStackedESN(training_input,...
                                            training_output,...
                                            sWin, sW, leakRate,...
                                            stdNoise, node_type, output_type);
            Y = genStackedESN(testing_input, test_data_size,  x, sWin, sW,...
                                sWout, leakRate, stdNoise, node_type,...
                                output_type, feedback_scaling );
            
            dif = Y - testing_output;
            % temporary to skip huge errors
            mse_temp = sum(sqrt(sum(dif.^2))); 
            if(mse_temp < 20)
                % calculating online the mean and the std of the MSE of each trial 
                successful_run_counter = successful_run_counter + 1;
                delta = (mse_temp - mse_mean);
                mse_mean = mse_mean + delta/successful_run_counter;
                mse_std =  mse_std + delta*(mse_temp - mse_mean);
            end
        end
        disp(['E(MSE)/E(STD): ', num2str(mse_mean, '%2.2f'), '/', num2str(sqrt(mse_std), '%2.2f')]);
        mse_results(experiment) = mse_mean;
        mse_results_std(experiment) = sqrt(mse_std / (successful_run_counter - 1));
    end

    [best_mse, best_experiment] = min(mse_results);

    best_paramters = parameters_grid(best_experiment,:);
end