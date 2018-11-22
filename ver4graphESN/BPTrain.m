classdef BPTrain < handle
    properties
    end
   
    methods
        function obj = BatchOutputLayerTrain()
        end
        %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        function x_collected = train(obj, net, input, target, initLen, type)
            bpRoute = net.getBackPropogationRoute();
            outputId = net.getIdByType('output');
            toOuputIDs = net.getPrevNodes(outputId);
            
           nExamples = size(target,2); 

            switch type
                case 'elman'


            end

            % put the trained weights in the weight matrix of the network 
             net.setWeightsForSelectedWeights(toOuputIDs, outputId, W_out);
      end
      %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      function error = mse(obj, x, y)
            dif = x - y;
            error = sqrt(trace(dif' * dif));
      end
      %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
    end
end