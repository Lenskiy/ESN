classdef Network < handle
    properties (SetAccess = private)
       networkArchitecture;
       W;
       blocksInds;
       x;
       layers;
       actFuncs;
       leakage;
       IDs;
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
            obj.IDs = [];
        end
        %------------------------------------------------------------------------------------
        function id = addLayer(obj,  numNodes, layerType, params)
            if(isempty(obj.layers))
                id = 1;
            else
                id = obj.layers{end}.id + 1;
            end
            obj.IDs = [obj.IDs id];
            ind = length(obj.IDs);
            obj.blocksInds(ind, :) = [size(obj.W,1) + 1, size(obj.W,1) + numNodes];
            obj.W(end+1:end+numNodes,end+1:end+numNodes) = 0;
            obj.layers{ind}  = (struct('id', id,...
                                    'numNodes', numNodes,...
                                    'layerType', layerType,... 
                                    'params', params));

            obj.x = [obj.x; zeros(numNodes, 1)];
            obj.leakage = [obj.leakage; params.leakage * ones(numNodes, 1)];
            obj.setActiveFn(ind, params.nodeType);
            obj.initLayer(ind);
            
            obj.networkArchitecture(ind, ind) = struct('type', [], 'connectivity', []);
        end
        %------------------------------------------------------------------------------------
        function removeLayer(obj,  id) % broken, lot of works needs to be done
                    ind = find(obj.IDs == id); % could be speeded up
                    obj.IDs(ind) = [];
                    obj.layers(ind:end - 1) = obj.layers(ind+1:end);
                    obj.layers(end) = [];
                    inds = obj.blocksInds(ind, :);
                    if(ind < size(obj.blocksInds,1))
                        obj.blocksInds(ind + 1, :) = obj.blocksInds(ind , 1) + obj.blocksInds(ind + 1:end, :) - obj.blocksInds(ind , 2) - 1;
                    end
                    obj.blocksInds(ind, :) = [];
                    obj.W(inds(1):inds(2), :) = [];
                    obj.W(:, inds(1):inds(2)) = [];
                    obj.x(inds(1):inds(2)) = [];
                    obj.leakage(inds(1):inds(2)) = [];
                    % remove connections in networkArchitecture
                    obj.networkArchitecture(ind, :) = [];
                    obj.networkArchitecture(:, ind) = [];
                    
        end
        %------------------------------------------------------------------------------------
        function initLayer(obj, id)
                switch obj.layers{id}.layerType
                    case 'input'
                        ;
                    case 'output'
                        ;
                    case 'reservoir'
                        obj.initReservoir(id);
                end
        end

        %------------------------------------------------------------------------------------
        function initReservoir(obj, id)
            ind = obj.IDs == id;
            blockSize = obj.blocksInds(ind,2) - obj.blocksInds(ind,1) + 1;
            %rng('shuffle');
            % generate connecting weight in reservoir
            switch (obj.layers{id}.params.initType)
                case 'rand'
                    W = sprand(blockSize, blockSize, obj.layers{id}.params.connectivity);
                    W(W ~= 0) = W(W ~= 0) - 0.5;
                case 'randn'
                %otherwise
                    W = sprandn(blockSize, blockSize, obj.layers{id}.params.connectivity);
            end
            obj.setWeights(ind, W);
            obj.changeRadiusTo(ind, obj.layers{ind}.params.radius);         
        end 
        %------------------------------------------------------------------------------------
        function changeLayerParamsTo(obj, id, params)
            ind = obj.IDs == id;
            names = fieldnames(params);
            for k = 1:length(names)
                switch names{k}
                    case 'nodeType'
                        obj.setActiveFn(ind, params.nodeType)
                    case 'radius'
                        obj.changeRadiusTo(ind, params.radius);
                    case  'leakage'
                        obj.changeLeakageTo(ind, params.leakage);
                    case 'connectivity'
                        ;
                    case 'initType'
                        obj.layers{id}.params.initType = params.initType;
                        obj.initLayer(id);
                        
                end
                
            end
        end
        %------------------------------------------------------------------------------------
        function changeConnectivityTo(obj, connectivity)
%             delta = obj.params.connectivity - connectivity;
%             if (delta > 0)
%                 inds = find(obj.W == 0);
%                 
%             else
%                 
%             end
        end
        %------------------------------------------------------------------------------------
        function changeLeakageTo(obj, ind, leakage)
            obj.layers{ind}.params.leakage = leakage;
            inds = obj.blocksInds(ind,1):obj.blocksInds(ind,2);
            obj.leakage(inds) = leakage;
        end
        %------------------------------------------------------------------------------------
        function setActiveFn(obj, ind, nodeType)
            obj.layers{ind}.params.nodeType = nodeType;
        	switch(nodeType)
                case 'linear'
                    obj.layers{ind}.actFunc = @(x) (x);
                case 'tanh'
                    obj.layers{ind}.actFunc = @(x) tanh(x);
            end
        end
        %------------------------------------------------------------------------------------
        function changeRadiusTo(obj, ind, radius)
            obj.layers{ind}.params.radius = radius;
            inds = obj.blocksInds(ind,1):obj.blocksInds(ind,2);
            opts.tol = 1e-3;
            maxVal = max(abs(eigs(obj.W(inds,inds), 1, 'lm', opts)));
            obj.W(inds, inds) = radius * obj.W(inds,inds)/maxVal;
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
            srcInd = find(obj.IDs == sourceId);
            dstInd = find(obj.IDs == destId);
            assert(~isempty(srcInd), ['ID: ', num2str(srcInd), ' is not found']);
            assert(~isempty(dstInd), ['ID: ', num2str(dstInd), ' is not found']);
            obj.networkArchitecture(srcInd, dstInd) = params;
            srcInds = obj.blocksInds(srcInd,1):obj.blocksInds(srcInd,2);
            destInds = obj.blocksInds(dstInd, 1):obj.blocksInds(dstInd, 2);
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
        function visualize(obj)
            conMat = obj.getConnectivityMatrix();
            graph = biograph(conMat);
            graph.ArrowSize = 5;

            for k = 1:length(obj.layers)
                if(strcmp(obj.layers{k}.layerType, 'input') || strcmp(obj.layers{k}.layerType,'output'))
                    graph.Nodes(obj.layers{k}.id).Color = [0.7 0.7 0.5];
                end
                if(strcmp(obj.layers{k}.layerType, 'reservoir'))
                    graph.Nodes(k).shape = 'circle';
                    graph.Nodes(k).LineWidth = 10 * obj.layers{k}.params.connectivity;
                    numIntStates = obj.blocksInds(end,end) -  obj.blocksInds(2,end); % *needs take into account deleted layers
                    graph.Nodes(k).Size = round(80 * [obj.layers{k}.numNodes; obj.layers{k}.numNodes]/numIntStates) + 20;
                end
                
                 graph.Nodes(k).Label = [num2str(obj.layers{k}.id), ':',obj.layers{k}.layerType];
                
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
            ind = find(obj.layers{k}.id == obj.IDs); %% could be speeded up
            inds = obj.blocksInds(ind,1):obj.blocksInds(ind,2);
            x(inds) = obj.layers{k}.actFunc(x(inds)); %this part a bit slows down
        end
      end
      %-------------------------------------------------------------------------------------
      function setWeights(obj, ind, W)
        inds = obj.blocksInds(ind,1):obj.blocksInds(ind,2);
        obj.W(inds, inds) = W;
      end
      %-------------------------------------------------------------------------------------
      function W = getWeights(obj, ind)
        inds = obj.blocksInds(ind,1):obj.blocksInds(ind,2);
        W = obj.W(inds, inds); 
      end
     %-------------------------------------------------------------------------------------
      function x = forward(obj, u) % Calculates next step of the system
          obj.setInput(u);
          obj.x = (1 - obj.leakage) .* obj.activate(obj.W * obj.x) + obj.leakage .* obj.x;
          x = obj.x;
      end
     %-------------------------------------------------------------------------------------
      function y = predict(obj, input) 
        y = zeros(length(input), 1);
        for k = 1 : length(input)
            obj.forward(input(k));
            y(k) = obj.getOutput();
        end
      end    
     %-------------------------------------------------------------------------------------
	function y = generate(obj, input, genLenght)
        if(nargin() == 2)
            genLenght = length(input);
        end
        obj.forward(input(1));
        y = zeros(genLenght, 1);
        y(1) = obj.getOutput();
        for k = 2 : genLenght
            obj.forward(y(k - 1));
            y(k) = obj.getOutput();
        end
      end  
    end
end