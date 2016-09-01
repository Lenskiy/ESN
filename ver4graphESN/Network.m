classdef Network < handle
    properties (SetAccess = private)
       networkArchitecture;
       W;
       blocksInds;
       x;
       layers;
       actFuncs
    end
    
    properties
    end
    % add methods to work with linked links
    methods
        function obj = Network()
            networkArchitecture(1,1) = struct('type', [], 'connectivity', []);
            obj.networkArchitecture = networkArchitecture;
            obj.W = sparse([]);
            obj.x = [];
            obj.layers = [];
        end
        %------------------------------------------------------------------------------------
        function id = addLayer(obj,  numNodes, layerType, params)
            if(isempty(obj.layers))
                id = 1;
            else
                id = obj.layers{end}.id + 1;
            end
            obj.blocksInds(id, :) = [size(obj.W,1) + 1, size(obj.W,1) + numNodes];
            obj.W(end+1:end+numNodes,end+1:end+numNodes) = 0;
            obj.layers{id}  = (struct('id', id,...
                                    'numNodes', numNodes,...
                                    'layerType', layerType,... 
                                    'params', params));

            obj.x = [obj.x; zeros(numNodes, 1)];
            switch(params.node)
                case 'linear'
                    obj.layers{id}.actFunc = @(x) (x);
                case 'tanh'
                    obj.layers{id}.actFunc = @(x) tanh(x);
            end
            obj.initLayer(id);
        end
        %------------------------------------------------------------------------------------
        function removeLayer(obj,  id) % broken, lot of works needs to be done
            for k = 1:length(obj.layers)
                if(obj.layers{k}.id == id)
                    obj.layers(k:end - 1) = obj.layers(k+1:end);
                    obj.layers(end) = [];
                    inds = obj.blocksInds(id, :);
                    obj.blocksInds(id, :) = [0, 0];
                    obj.W(inds, :) = 0;
                    obj.W(:, inds) = 0;
                    % remove connections in networkArchitecture
                    obj.networkArchitecture(id, :) = struct('type', [], 'connectivity', []);
                    obj.networkArchitecture(:, id) = struct('type', [], 'connectivity', []);
                    break;
                end
            end
        end
        %------------------------------------------------------------------------------------
        function setConnections(obj, arch, type)
            [src, dst]= find(arch ~= 0);
            for k = 1:length(src);
                    params = struct('type', type, 'connectivity', arch(src(k), dst(k)));
                    obj.setConnection(src(k), dst(k), params);
            end
        end
        %------------------------------------------------------------------------------------
        function setConnection(obj, sourceId, destId, params)
            obj.networkArchitecture(sourceId, destId) = params;
            srcInds = obj.blocksInds(sourceId,1):obj.blocksInds(sourceId,2);
            destInds = obj.blocksInds(destId,1):obj.blocksInds(destId,2);
            switch params.type
                case 'rand'
                    W = sprand(length(srcInds), length(destInds), params.connectivity);
                    W(W ~= 0) = W(W ~= 0) - 0.5;
                case 'randn'
                    W = sprandn(length(srcInds), length(destInds), params.connectivity);
            end
            obj.W(destInds, srcInds) = W;
        end
        %------------------------------------------------------------------------------------
        function initLayer(obj, layerId)
                switch obj.layers{layerId}.layerType
                    case 'input'
                        ;
                    case 'output'
                        ;
                    case 'reservoir'
                        obj.initReservoir(layerId);
                end
        end

        %------------------------------------------------------------------------------------
        function initReservoir(obj, layerId)
            inds = obj.blocksInds(layerId,1):obj.blocksInds(layerId,2);
            %rng('shuffle');
            % generate connecting weight in reservoir
            switch (obj.layers{layerId}.params.init_type)
                case 'rand'
                    W = sprand(length(inds), length(inds), obj.layers{layerId}.params.connectivity);
                    W(W ~= 0) = W(W ~= 0) - 0.5;
                case 'randn'
                %otherwise
                    W = sprandn(length(inds), length(inds), obj.layers{layerId}.params.connectivity);
            end
            % compute spectral radius i.e. the largest absolute eigen value 
            opts.tol = 1e-3;
            maxVal = max(abs(eigs(W, 1, 'lm', opts)));
            if(maxVal ~= 0)
                obj.W(inds, inds) = obj.layers{layerId}.params.radius * W/maxVal; % normalize W
            end            
        end 
        %------------------------------------------------------------------------------------
        function visualize(obj)
            conMat = obj.getConnectivityMatrix();
            graph = biograph(conMat);
            graph.ArrowSize = 5;
            for k = 1:length(obj.layers)
                if(strcmp(obj.layers{k}.layerType, 'input') || strcmp(obj.layers{k}.layerType,'output'))
                    graph.Nodes(obj.layers{k}.id).Color = [0.7 0.7 0.5];
                end
                if(strcmp(obj.layers{k}.layerType, 'reservoir'))
                    graph.Nodes(obj.layers{k}.id).shape = 'circle';
                    graph.Nodes(obj.layers{k}.id).LineWidth = 10 * obj.layers{k}.params.connectivity;
                    numIntStates = obj.blocksInds(end,end) -  obj.blocksInds(2,end);
                    graph.Nodes(obj.layers{k}.id).Size = round(80 * [obj.layers{k}.numNodes; obj.layers{k}.numNodes]/numIntStates) + 20;
                end
                
                 graph.Nodes(obj.layers{k}.id).Label = [num2str(obj.layers{k}.id), ':',obj.layers{k}.layerType];
                
            end
            
            for k = 1:length(graph.Edges)
                graph.Edges(k).LineWidth = 10 * graph.Edges(k).Weight;
            end
            
            h = view(graph);
        end
        %------------------------------------------------------------------------------------
        function mat = getConnectivityMatrix(obj)
            dim = size(obj.networkArchitecture, 1);
            mat = sparse(dim, dim);
            for k = 1:dim
                for l = 1:dim
                    if(~isempty(obj.networkArchitecture(k,l).connectivity))
                        mat(k, l) = obj.networkArchitecture(k,l).connectivity;
                    end
                end
            end
        end
      %-------------------------------------------------------------------------------------
      function x = forward(obj, u) % Calculates next step of the system
          obj.setInput(u);
          obj.x = obj.activate(obj.W * obj.x);
          x = obj.x;
      end
      %-------------------------------------------------------------------------------------
      function y = getOutput(obj)
            y = obj.x(obj.blocksInds(2, 1):obj.blocksInds(2, 2));
      end
      %-------------------------------------------------------------------------------------
      function setInput(obj, u)
        obj.x(obj.blocksInds(1, 1):obj.blocksInds(1, 2)) = u;
      end
      %-------------------------------------------------------------------------------------
      function x = activate(obj, x)
        for k = 1:length(obj.layers)
            id = obj.layers{k}.id;
            inds = obj.blocksInds(id,1):obj.blocksInds(id,2);
            x(inds) = obj.layers{k}.actFunc(x(inds)); %this part a bit slows down
        end
      end
      %-------------------------------------------------------------------------------------
      function W = getWeights(id)
          W = [];
          for k = 1:length(obj.layers)
            if(obj.layers{k}.id == id)
                W = obj.W(obj.blocksInds(id,1):obj.blocksInds(id,2), :);
                break;
            end
          end
          
      end
    end
end