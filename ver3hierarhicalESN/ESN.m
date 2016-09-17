classdef ESN < handle
   properties
        reservoir;
        architecture = struct('inputDim',   0, ...
                              'numNodes',   0, ... 
                              'outputDim',  0);      
        parameters  = struct( 'node_type',     'tanh',...
                              'radius',      0, ...
                              'leakage',    0, ... 
                              'connectivity',0,...
                              'init_type', 'rand');            
        W_in;
        W_fb;
        W_out;
        Y_last;
        nodes;
   end
   %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   methods   
      function obj = ESN(architecture, parameters)
          obj.architecture = architecture;
          obj.parameters = parameters;
          %obj.nodes = repmat(Neuron(parameters.neuron), architecture.numNodes, 1 );
          obj.reservoir = Reservoir(architecture.numNodes, parameters);
          obj.initialize(parameters.init_type);
          obj.Y_last = 0;
      end 
      %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      function initialize(obj, init_type)
            rng('shuffle');
            switch (init_type)
                case 'const'
                    obj.W_in = ones(obj.architecture.numNodes,...
                        1 + obj.architecture.inputDim) / obj.architecture.numNodes;
                case 'rand'
                    obj.W_in = (rand(obj.architecture.numNodes,...
                        1 + obj.architecture.inputDim) - 0.5);
                    %obj.W_fb = (rand(obj.architecture.numNodes,...
                    %    obj.architecture.outputDim) - 0.5);
                case 'randn'
                    obj.W_in = (randn(obj.architecture.numNodes,...
                        1 + obj.architecture.inputDim));
            end
            obj.Y_last = 0;
            obj.W_fb = zeros(obj.architecture.numNodes, obj.architecture.outputDim);
      end
      %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      function X_cur = forward(obj, u)
            
          %  X_cur = obj.reservoir.forward(obj.W_in * [1; u] + obj.W_fb * obj.Y_last);
           X_cur = obj.reservoir.forward(obj.W_in * [1; u]);
      end
      %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      function y = evaluate(obj, u)
            X = forward(obj, u);
            obj.Y_last =  obj.W_out * [1; u; X];     
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
            Y_last = 0;
            obj.reservoir.setInitStates();
      end
      %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      function resetInitStates(obj)
            Y_last = 0;
            obj.reservoir.resetInitStates();
      end
	 %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      function error = backward(obj, x)

      end
   end
end
   