function elmanErros = trainElman(trainingData, learning_rate, netSize, N_trials)

    dataLen = size(trainingData,2);

    eras_count = 500;
    inpSize = 3;
    train_size = dataLen * 0.8;
    test_size = dataLen - train_size;
    seq_length = 320; % Caution, seq_length, hprev and eras_count are strongly dependent on each other!!!
    %Nsize = [4     15       36      65     102     147       200      261       330];
    elmanErros = zeros(N_trials, length(netSize));  
    
    for l = 1:length(netSize)
        %l
        for k = 1:N_trials
            %k
            n = 0;
            p = 1;
            Wxh = rand(netSize(l), inpSize) * 0.01;
            Whh = rand(netSize(l), netSize(l)) * 0.01;
            Why = rand(inpSize, netSize(l)) * 0.01;
            bh = zeros(netSize(l), 1);
            by = zeros(inpSize, 1);

            mWxh = zeros(netSize(l), inpSize);
            mWhh = zeros(netSize(l), netSize(l));
            mWhy = zeros(inpSize, netSize(l));
            mbh = zeros(netSize(l), 1);
            mby = zeros(inpSize, 1);

            hprev = zeros(netSize(l), 1);

            while n < eras_count
                % prepare inputs (we're sweeping from left to right in steps seq_length long)
                if (p + seq_length > train_size + 1 || n == 0)
                    hprev = zeros(netSize(l), 1);  % reset RNN memory
                    p = 1;  % go from start of data
                end
                

                [~, dWxh, dWhh, dWhy, dbh, dby, hprev] = lossFun(trainingData(:, p:p+seq_length-1), trainingData(:, (p+1):p+seq_length), hprev, Wxh, Whh, Why, bh, by, netSize(l), n, eras_count);
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
                n = n + 1;  % iteration counter

            end
            [~, sse_] = sample(trainingData(:, train_size:dataLen-1), trainingData(:, train_size+1:dataLen), hprev, Wxh, Whh, Why, bh, by);
            fprintf('%d, %s\n', netSize(l), num2str( sse_ ));
            elmanErros(l,k) = sse_;
        end
        
    end
    
end
    
function [Out, mse]=sample(test_inp, test_targ, hState, Wxh, Whh, Why, bh, by)
    %sample a sequence of integers from the model
    %h is memory state, seed_ix is seed letter for first time step
    testLength =  size(test_inp, 2); 
    Out = zeros(3, testLength);
    %u = test_inp(1,:)';
    for t = 1:testLength
        hState = tanh(Wxh * test_inp(:,t) + Whh * hState + bh);
        Out(:, t) = Why*hState + by;
        %u = Out(t,:)';
    end
    mse = sum(sum((Out - test_targ).^2))/testLength;
end

function [loss, dWxh, dWhh, dWhy, dbh, dby, prevSt] = lossFun(xs, targ_out, hprev, Wxh, Whh, Why, bh, by, Nsize, n, e)
    inpDim = size(xs,1);
    dataLength = size(xs,2);
    hs = zeros(Nsize(l), dataLength);
    ys = zeros(inpDim, dataLength);

    hs(:, 1) = hprev;
    
    % start forward pass
    for t = 1:dataLength
       hs(:, t + 1) = tanh(Wxh*xs(:, t) + Whh*hs(:, t) + bh); % actually, hs{t,1} is previos hs
    end
     
    ys = Why*hs(:, 2:end) + by;
    delta = (ys - targ_out).^2;
    
    loss = sum(delta)./dataLength;
    
    % start backprop pass
    dWxh = zeros(Nsize(l), inpDim);
    dWhh = zeros(Nsize(l), Nsize(l));
    dWhy = zeros(inpDim, Nsize(l));
    dbh = zeros(Nsize(l), 1);
    dby = zeros(inpDim, 1);
    
    dhnext = zeros(Nsize(l), 1);
    
    for t = dataLength:-1:1
        dy = 2*(ys(:,t) - targ_out(:, t));
        dWhy = dWhy + 2*dy*hs(:, t+1)';
        dby = dby + dy;
        dh = (Why') * dy + dhnext;
        dhraw = (1 - hs(:, t + 1) .* hs(:, t + 1)) .* dh; % actually, hs{t,1} is previos hs
        dbh = dbh + dhraw;
        dWxh = dWxh + dhraw * xs(:, t)'; 
        dWhh = dWhh + dhraw * hs(:, t)';
        dhnext = Whh'*dhraw;
    end     
        
    % clip params to mitigate exploding gradients
    dWxh = max(min(dWxh,1),-1);
    dWhh = max(min(dWhh,1),-1);
    dWhy = max(min(dWhy,1),-1);
    dbh  = max(min(dbh,1),-1);
    dby  = max(min(dby,1),-1);
    if (n < e-1)
        prevSt = hs(:, dataLength+1);
    else
        if (n == e-1)
            prevSt = hs(:, dataLength);
        end
    end     
end