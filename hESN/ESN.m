classdef ESN < handle
   properties
        reservoir;
        numNodes;
        input_size;
        W_in;
        W_fb;
        W_out;
        X_init
   end
   %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   methods
      function obj = ESN(type, numNodes, input_size, radius, leakage, connectivity, init_type)
          obj.numNodes = numNodes;
          obj.input_size = input_size;
          obj.reservoir = Reservoir(type, numNodes, radius, leakage, connectivity);
          obj.initialize(init_type);
      end 
      %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      function initialize(obj, init_type)
            rng('shuffle');
            switch (init_type)
                case 'const'
                    obj.W_in = ones(obj.numNodes, 1 + obj.input_size) / obj.numNodes;
                case 'rand'
                    obj.W_in = (rand(obj.numNodes,1 + obj.input_size) - 0.5);
                case 'randn'
                    obj.W_in = (randn(obj.numNodes,1 + obj.input_size));
            end
      end
      %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      function y = forward(obj, u)
            X_cur = obj.reservoir.forward(obj.W_in * [1; u]);
            y =  obj.W_out * [X_cur; 1; u];           
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
   