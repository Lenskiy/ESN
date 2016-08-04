classdef StackedESN < handle
    properties (SetAccess = private)
        X_temp;
        totalStates;
    end
    properties
        
        architecture;% = struct('inputDim',   0, ...
                     %         'numNodes',   [0], ... 
                     %         'outputDim',  0);      
        parameters;%  = struct( 'node_type',     'tanh',...
                   %           'radius',      0, ...
                   %           'leakage',    0, ... 
                   %           'connectivity',0,...
                   %           'init_type', 'rand');   
        reservoirs;
        
        W_in;      % Input weight matrix for all reservoirs
        W_fb;      % A cell array of fee
        W_out;
        Y_last
    end
   
    methods
      function obj = StackedESN(sArchitecture, sParameters, init_type)
            obj.architecture = sArchitecture;
            obj.parameters = sParameters;
            obj.reservoirs = Reservoir(sArchitecture.numNodes, sParameters);

            obj.totalStates = sum(sArchitecture.numNodes);
            obj.X_temp = zeros(obj.totalStates,1);
            obj.initialize(init_type);
      end
      %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      function initialize(obj, init_type)
            rng('shuffle');
            for k = 1:length(obj.architecture.numNodes)
                switch (init_type)
                    case 'const'
                        obj.W_in{k} = ones(obj.architecture.numNodes(k),...
                            1 + obj.architecture.inputDim) / obj.architecture.numNodes;
                    case 'rand'
                        obj.W_in{k} = (rand(obj.architecture.numNodes(k),...
                            1 + obj.architecture.inputDim) - 0.5);
                        %obj.W_fb = (rand(obj.architecture.numNodes,...
                        %    obj.architecture.outputDim) - 0.5);
                    case 'randn'
                        obj.W_in{k} = (randn(obj.architecture.numNodes(k),...
                            1 + obj.architecture.inputDim));
                end
                obj.W_fb{k} = zeros(obj.architecture.numNodes(k), obj.architecture.outputDim);
            end
            obj.Y_last = 0;
      end
      %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      function X = forward(obj, u) % Calculates next step of the system
          index = 0;
          for k = 1:length(obj.architecture.numNodes) % collect states from all reservoirs
             obj.X_temp(index+1:index + obj.architecture.numNodes(k)) =...
                 obj.reservoirs(k).forward(obj.W_in{k} * [1; u] + obj.W_fb{k} * obj.Y_last);
             index = index + obj.architecture.numNodes(k);
          end
          X = obj.X_temp;
      end
      %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      function y = evaluate(obj, u) % Evaluates output the system
            X = forward(obj, u);
            obj.Y_last =  obj.W_out * [X; 1; u];     
            y = obj.Y_last;
      end
      %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      function Y = generate(obj, input, gen_length, feedback_scaling)
          Y = zeros(size(input,1), gen_length);
          Y(:, 1) = feedback_scaling * evaluate(obj, input(:, 1));
          for k = 2:gen_length
               Y(:, k) = feedback_scaling * evaluate(obj, Y(:, k - 1));
          end
      end
      %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      function Y = predict(obj, input, feedback_scaling)
          Y = zeros(size(input,1), size(input,2));
          for k = 1:size(input,2)
               Y(:, k) = feedback_scaling * evaluate(obj, input(:, k));
          end
      end
      %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      function setInitStates(obj)
          for k = 1:length(obj.architecture.numNodes)
            obj.reservoirs(k).setInitStates();
          end
      end
      %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      function resetInitStates(obj)
          for k = 1:length(obj.architecture.numNodes)
            obj.reservoirs(k).resetInitStates();
          end
      end
      %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      function error = backward(obj, x)

      end       
    end
end