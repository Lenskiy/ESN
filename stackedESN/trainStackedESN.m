function [Wout, x, x_evolution_only, tf] = trainStackedESN(input,...
                                                        target, Win, W, lr, sigma, node_type, output_type)
    trainLen = size(input,1);
    initLen = round(0.1 * trainLen);
    input_size = size(input, 2);
    total_states_size = 0;
    
    switch(node_type)
        case 'tanh'
            node_type = 1;
        case 'logistic'
            node_type = 2;
        case 'relu'
            node_type = 3;
        case 'linear'
            node_type = 4;
        otherwise
            node_type = 1;
    end
    
    % allocated memory for the design (collected states) matrix
    for k = 1:length(W)
        total_states_size = total_states_size + size(W{k},1);
    end    

    x_evolution = zeros(1 + input_size + total_states_size, trainLen - initLen);
    % run the reservoir with the data and collect states

    cur_state_ind = input_size + 1 + 1; % We skip first columns, init later
    for k = 1:length(W)
        
        Win_k = Win{k};
        W_k = W{k};
        x_k = zeros(size(W_k,1),1);
        numStates_k = length(x_k);
        for j = 1:trainLen
            u = input(j, :)';

            switch (node_type)
                case 1 %Hypebolic tangent
                x_updated =  tanh( Win_k * [1; u] + W_k * x_k ) + sigma * randn(size(x_k,1), 1);
                case 2 %Relu
                x_updated =  max( Win_k * [1; u] + W_k * x_k, 0) + sigma * randn(size(x_k,1), 1);
                case 3 %logistic
                x_updated =  tanh( Win_k * [1; u] + W_k * x_k)/2 + 0.5 + sigma * randn(size(x_k,1), 1); 
                case 4 %linear
                x_updated =  (Win_k * [1; u] + W_k * x_k) + sigma * randn(size(x_k,1), 1);
            end
            
            x_k = (1 - lr(k)) * x_k + lr(k) * x_updated;
            
            if j > initLen % don't store a first few input states 
                x_evolution(cur_state_ind:cur_state_ind + numStates_k - 1, j - initLen) = x_k;
            end
        end
        x{k} = x_k;
        cur_state_ind = cur_state_ind + numStates_k;
    end
    
    % Initialzie first columns with bais and input values
    x_evolution(1:input_size + 1, : ) = [ones(1, trainLen - initLen); input(initLen + 1:end,:)']; 
    % skip the bias and input vector
	x_evolution_only = x_evolution(1 + input_size + 1:end, :);
  	switch output_type
        case 'linear'
            tf = eye(size(x_evolution,1), size(x_evolution,1));
        case 'quadratic'
            %pre-locate memory
            x_evolution_plus = zeros((total_states_size - 1) * total_states_size / 2 + 2 * total_states_size + input_size + 1, size(x_evolution_only,2));

            for i = 1:size(x_evolution_only,2)
                X = nchoosek(x_evolution_only(:,i),2);
                X_ = X(:,1).*X(:,2);
                % include bais, inputs, ESN states, squared ESN states, and producs of
                % all combinations of states: states = {x,y}, then we have {1, u, x, y, x^2, y^2, xy}
                x_evolution_plus(:, i) = [x_evolution(:, i); x_evolution_only(:,i).^2; X_];
            end
            x_evolution = x_evolution_plus;
            tf = eye(size(x_evolution,1), size(x_evolution,1));
        case 'pca'
            %m = mean(x_evolution)';
            %Xc = x_evolution'-repmat(m,1,size(x_evolution,1)); % Center and transpose

             %Obtain covariance matrix and calculate egien vectors and values
            CovMat = x_evolution * x_evolution';
            %Calculate eigen values and vectors
            [evec eval] = eig(CovMat);
            %Sort values in descending order
            %Eigenvectors that correspond to larger eigenavalues will be used for
            %dimensionality reduction
            tf = evec(end-9:end,:);
            x_evolution = tf * x_evolution;
    end
    
   
    % Train ouput layer by solving a overdetermined system of linear equations 
    %Xinv = pseudoinverse(x_evolution, [],'lsqr');
    %--------------
    Xinv = pseudoinverse(x_evolution,[],'lsqr', 'tikhonov', {@(x,r) r*normest(x_evolution)*x, 1e-4});
    %-------------
    Wout =  target(initLen + 1:end, :)' * Xinv; 
    
    % Add here the Wiener-Hopf solution
end