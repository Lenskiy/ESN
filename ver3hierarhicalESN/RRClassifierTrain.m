classdef RRClassifierTrain < handle
    properties
    end
   
    methods
        function obj = RRClassifierTrain()
        end
        %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        function [error] = train(obj, esn, input, target, classes)
            
            
            % For each input row element, calc time average activation
            % WARNING: WORKS ONLY FOR SINGLE INPUT CURRENTLY!
            S = [];
            T = [];
            
            esn.setInitStates();
            
            for r = 1:size(input,1)
                
                esn.resetInitStates();
               
                X = zeros(sum(esn.architecture.numNodes) + esn.architecture.inputDim + 1, size(target,2));
                
                for j = 1:size(input(r,:), 2)
                    u = input(r, j); 
                    X(1:sum(esn.architecture.numNodes), j) = esn.forward(u);
                end
            
                X(sum(esn.architecture.numNodes) + 1:end, :) = [ones(1, size(input,2)); input(r,:)];
                
                % Time average
                M = mean(X,2);
            
                S = [S M];
              
                % Add Target vector
                t = zeros(classes,1);
                t(target(r)+1,1) = 1;
                              
                T = [T t];
            end
            
            Sinv =  S' * inv(S*S' + 0.001*eye(size(S,1)));
          
            esn.W_out =   T * Sinv;
   
            % Calc classification errors
          
            P = esn.W_out*S;
            
            h=0;
            
            for i = 1:size(T,2)
                [maxval, maxind] = max(P(:,i));  
                
                if maxind-1 == target(i)
                    h = h +1;
                end
            end
         
            % Hit rate
            h/size(target,1)
            
        end
      %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      
      
      
      
    end
end