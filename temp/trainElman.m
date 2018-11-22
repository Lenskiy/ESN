function elmanErros = trainElman(trainingData, learning_rate, netSize, N_trials)
    %delete(gcp('nocreate'));
    %parpool('local',4);
    
    dataLen = size(trainingData, 2);
    inpSize = size(trainingData, 1);
    
    numEpochs = 50000;
    trainSize = dataLen * 0.8;
    seq_length = 320; % Caution, seq_length, hprev and eras_count are strongly dependent on each other!!!
    elmanErros = zeros(N_trials, length(netSize));  
    
    for l = 1:length(netSize)
        %l
        hidSize = netSize(l);
        for k = 1:N_trials
            %k
            epoch = 0;
            p = 1;
            Wxh = rand(hidSize, inpSize) * 0.01;
            Whh = rand(hidSize, hidSize) * 0.01;
            Why = rand(inpSize, hidSize) * 0.01;
            bh = zeros(hidSize, 1);
            by = zeros(inpSize, 1);

            mWxh = zeros(hidSize, inpSize);
            mWhh = zeros(hidSize, hidSize);
            mWhy = zeros(inpSize, hidSize);
            mbh = zeros(hidSize, 1);
            mby = zeros(inpSize, 1);

            hprev = zeros(hidSize, 1);

            while epoch < numEpochs
                % prepare inputs (we're sweeping from left to right in steps seq_length long)
                if (p + seq_length > trainSize + 1 || epoch == 0)
                    hprev = zeros(hidSize, 1);  % reset RNN memory
                    p = 1;  % go from start of data
                end
                

                [~, dWxh, dWhh, dWhy, dbh, dby, hprev] = lossFun(trainingData(:, p:p+seq_length-1), trainingData(:, (p+1):p+seq_length), hprev, Wxh, Whh, Why, bh, by, hidSize, epoch, numEpochs);
                %if (rem(n,1000) == 0)
                    %fprintf('iter %d, MSE: %s\n', n, num2str( loss ));  % print progress
                %end

                % perform parameter update with Adagrad
                %for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],
                %                              [dWxh, dWhh, dWhy, dbh, dby],
                %                              [mWxh, mWhh, mWhy, mbh, mby]):
                mWxh = mWxh + dWxh.*dWxh;
                Wxh = Wxh + (-learning_rate * dWxh ./ sqrt(mWxh + 1e-8)); % adagrad update

                mWhh = mWhh + dWhh.*dWhh;
                Whh = Whh + (-learning_rate * dWhh ./ sqrt(mWhh + 1e-8));

                mWhy = mWhy + dWhy.*dWhy;
                Why = Why + (-learning_rate * dWhy ./ sqrt(mWhy + 1e-8));

                mbh = mbh + dbh.*dbh;
                bh = bh + (-learning_rate * dbh ./ sqrt(mbh + 1e-8));

                mby = mby + dby.*dby;
                by = by + (-learning_rate * dby ./ sqrt(mby + 1e-8));

                p = p + seq_length;  % move data pointer
                epoch = epoch + 1;  % iteration counter

            end
            [~, sse_] = sample(trainingData(:, trainSize:dataLen-1), trainingData(:, trainSize+1:dataLen), hprev, Wxh, Whh, Why, bh, by);
            fprintf('%d, %s\n', hidSize, num2str( sse_ ));
            elmanErros(l,k) = sse_;
        end
        
    end
    
end
    
function [Out, mse]=sample(test_inp, test_targ, hState, Wxh, Whh, Why, bh, by)
    %sample a sequence of integers from the model
    %h is memory state, seed_ix is seed letter for first time step
    testLength =  size(test_inp, 2); 
    Out = zeros(3, testLength);
    for t = 1:testLength
        hState = tanh(Wxh * test_inp(:,t) + Whh * hState + bh);
        Out(:, t) = Why*hState + by;
    end
    mse = sum(sum((Out - test_targ).^2))/testLength;
end

function [loss, dWxh, dWhh, dWhy, dbh, dby, prevSt] = lossFun(xs, targ_out, hprev, Wxh, Whh, Why, bh, by, netSize, epoch, numEpochs)
    inpDim = size(xs,1);
    batchLength = size(xs,2);
    hs = zeros(netSize, batchLength);
    ys = zeros(inpDim, batchLength);
    hs(:, 1) = hprev;
    
    % start forward pass
    for t = 1:batchLength
       hs(:, t + 1) = tanh(Wxh * xs(:, t) + Whh * hs(:, t) + bh); 
       ys(:, t) = Why*hs(:, t + 1) + by;
    end
     
    
    delta = (ys - targ_out);
    
    loss = sum(delta.^2)./batchLength;
    
    % start backprop pass
    dWxh = zeros(netSize, inpDim);
    dWhh = zeros(netSize, netSize);
    dWhy = zeros(inpDim, netSize);
    dbh = zeros(netSize, 1);
    dby = zeros(inpDim, 1);
    
    dhnext = zeros(netSize, 1);
    for t = batchLength :-1:1     
        dWhy = dWhy + delta(:,t)*hs(:, t)';
        dby  = dby  + delta(:,t);
        dh   = (Why') * delta(:,t) + dhnext;
        dhraw = (1 - hs(:, t + 1).^2) .* dh; % 
        dbh = dbh + dhraw;
        dWxh = dWxh + dhraw * xs(:, t)'; 
        dWhh = dWhh + dhraw * hs(:, t)';
        dhnext = Whh'*dhraw;
    end     
        
    [mean(mean(dhraw))]
    
    % clip params to mitigate exploding gradients
    dWxh = max(min(dWxh, 5), -5);
    dWhh = max(min(dWhh, 5), -5);
    dWhy = max(min(dWhy, 5), -5);
    dbh  = max(min(dbh,  5), -5);
    dby  = max(min(dby,  5), -5);
    
     
    
    prevSt = hs(:, batchLength + 1);
    
%     if (epoch < numEpochs-1)
%         prevSt = hs(:, batchLength);
%     else
%         if (epoch == numEpochs-1)
%             prevSt = hs(:, batchLength - 1);
%         end
%     end     
end