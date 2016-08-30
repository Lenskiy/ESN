function [errMap, X, Y] = getErrorMap(parameters_grid, mse_results, sParams)
    % select indices of the set of parameters that correspond to the fixed
    % parameters
    allESNparameters = [];
    for k = 1:length(sParams)
        allESNparameters = [allESNparameters sParams{k}];
    end
    
    allESNparameters = cell2mat(allESNparameters);
    ind = 1:size(parameters_grid,1);
    for k = 1:length(allESNparameters)
        if (allESNparameters(k) ~= inf)
            ind_ = find(parameters_grid(ind, k) == allESNparameters(k));
            ind = ind(ind_);
        end
    end
    % select two parameters that will be used in ErrorMap
    paramsOfInterest = find(allESNparameters  == inf);

    errMap = reshape(mse_results(ind),...
        length(unique(parameters_grid(ind, paramsOfInterest(1)))),...
        length(unique(parameters_grid(ind, paramsOfInterest(2)))));
    
    [X, Y] = meshgrid(unique(parameters_grid(ind, paramsOfInterest(1))),...
        unique(parameters_grid(ind, paramsOfInterest(2))));
end