classdef HESN < handle
    properties
        esns;
        reservoir;
        architectures;
        parameters;
    end
   
    methods
      function obj = HESN(hArchitecture, hParameters, parameters)
            for k = 1:min(size(architectures,2), size(parameters,2))
                obj.esns(k) = esn(architectures(k), parameters(k));
                Reservoir(nodes, parameters)
            end
            
            
            
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
      function y = forward(obj, u)
            X_cur = obj.reservoir.forward(obj.W_in * [1; u] + obj.W_fb * obj.Y_last);
            obj.Y_last =  obj.W_out * [X_cur; 1; u];     
            y = obj.Y_last;
      end      
      %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      function Y = generate(obj, input, gen_length, feedback_scaling)
          Y = zeros(gen_length, size(input,2));
          Y(1, :) = feedback_scaling * forward(obj, input(1, :));
          for k = 2:gen_length
               Y(k, :) = feedback_scaling * forward(obj, Y(k - 1, :));
          end
      end
      %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      function Y = predict(obj, input, feedback_scaling)
          Y = zeros(size(input,2), size(input,1));
          for k = 1:size(u,2)
               Y(k, :) = feedback_scaling * forward(obj, input(k, :));
          end
      end      
      %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      function error = backward(obj, x)

      end        
    end
end