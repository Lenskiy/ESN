classdef BatchTrainClassifierOutputLayer < handle
    properties
    end
   
    methods
        function obj = BatchTrainClassifierOutputLayer()
        end
        %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        function [error] = train(obj, net, input, target, initLen)
            
            %assert(size(input, 3) == size(target, 2), 'Number of samples in input and ouput should be equal')
            
            nExamples = size(input,3);
            nSamplesInExample = size(input,2);
            
            outputId = net.getIdByName('output');
            toOuputIDs = net.getPrevNodes(outputId);
                
            avgStates = zeros(net.getNumberOfStates(toOuputIDs), nExamples);
            
            net.rememberStates();
            
            interestStates = zeros(net.getNumberOfStates(toOuputIDs), nSamplesInExample);
            
            for k = 1:nExamples
                if(mod(k, 100) == 0) 
                    k/nExamples
                end
                net.recallStates();
               
                for j = 1:nSamplesInExample
                    net.forward(input(:, j, k));
                    interestStates(:, j) = net.getStates(toOuputIDs);
                end          
                
                % Spatial average            
                avgStates(:, k) = mean(interestStates(:, initLen + 1:end),2);
            end
            
            
            Sinv =  avgStates' * inv(avgStates*avgStates' + 0.001*eye(size(avgStates,1)));

            W_out =   target * Sinv;
            net.setWeightsForSelectedWeights(toOuputIDs, outputId, W_out');
           
            % Calc classification errors
          
            P = W_out*avgStates;
            h=0;
            
            indTarget = vec2ind(target);
            for i = 1:nExamples
                [~, maxind] = max(P(:,i));  
                
                if maxind == indTarget(i)
                    h = h +1;
                end
            end
         
            % Hit rate
            error = 1 - h/nExamples;
            
        end
      %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    end
end