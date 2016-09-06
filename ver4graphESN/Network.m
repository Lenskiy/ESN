classdef Network < handle
    properties (SetAccess = private)
       networkArchitecture;
       W;
       blocksInds;
       x;
       layers;
       actFuncs;
       leakage;
       noiseStd
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
            obj.x = []; % the first index resereved for bias i.e. always 1;
            obj.leakage = [];
            obj.noiseStd = [];
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
                    ind = obj.idToInd(id); % could be speeded up
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
            layerIndex = obj.idToInd(id);
            switch obj.layers{layerIndex}.layerType
                    case 'input'
                        %obj.leakage(obj.getInds(id)) = 1;
                        obj.W(obj.getInds(id), obj.getInds(id)) = 1;
                    case 'output'
                        ;
                    case 'reservoir'
                        obj.initReservoir(id);
                    case 'bias'
                        obj.leakage(obj.getInds(id)) = 1;
                        obj.x(obj.getInds(id)) = 1;
                        obj.W(obj.getInds(id), obj.getInds(id)) = 1;
            end
        end

        %------------------------------------------------------------------------------------
        function initReservoir(obj, id)
            layerIndex = obj.idToInd(id);
            blockSize = obj.blocksInds(layerIndex,2) - obj.blocksInds(layerIndex,1) + 1; % make bocksInd part of layer structure
            %rng('shuffle');
            %generate connecting weight in reservoir
            switch (obj.layers{layerIndex}.params.initType)
                case 'rand'
                    W = sprand(blockSize, blockSize, obj.layers{layerIndex}.params.connectivity);
                    W(W ~= 0) = W(W ~= 0) - 0.5;
                case 'randn'
                %otherwise
                    W = sprandn(blockSize, blockSize, obj.layers{layerIndex}.params.connectivity);
            end
            obj.setWeights(id, id, W);
            obj.changeRadiusTo(id, obj.layers{layerIndex}.params.radius);         
        end 
        %------------------------------------------------------------------------------------
        function changeLayerParamsTo(obj, id, params)
            layerIndex = obj.idToInd(id);
            names = fieldnames(params);
            for k = 1:length(names)
                switch names{k}
                    case 'nodeType'
                        obj.setActiveFn(id, params.nodeType)
                    case 'radius'
                        obj.changeRadiusTo(id, params.radius);
                    case  'leakage'
                        obj.changeLeakageTo(id, params.leakage);
                    case 'connectivity'
                        ;
                    case 'initType'
                        obj.layers{layerIndex}.params.initType = params.initType;
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
        function changeLeakageTo(obj, id, leakage)
            layerIndex = obj.idToInd(id);
            obj.layers{layerIndex}.params.leakage = leakage;
            inds = obj.blocksInds(layerIndex,1):obj.blocksInds(layerIndex,2);
            obj.leakage(inds) = leakage;
        end
        %------------------------------------------------------------------------------------
        function setActiveFn(obj, id, nodeType)
            layerIndex = obj.idToInd(id);
            obj.layers{layerIndex}.params.nodeType = nodeType;
        	switch(nodeType)
                case 'linear'
                    obj.layers{layerIndex}.actFunc = @(x) (x);
                case 'tanh'
                    obj.layers{layerIndex}.actFunc = @(x) tanh(x);
                case 'lu'   
                    obj.layers{layerIndex}.actFunc = @(x) max(0, x);
            end
        end
        %------------------------------------------------------------------------------------
        function changeRadiusTo(obj, id, radius)
            layerIndex = obj.idToInd(id);
            obj.layers{layerIndex}.params.radius = radius;
            inds = obj.blocksInds(layerIndex,1):obj.blocksInds(layerIndex,2);
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
            srcInd = obj.idToInd(sourceId);
            dstInd = obj.idToInd(destId);
            assert(~isempty(srcInd), ['ID: ', num2str(srcInd), ' is not found']);
            assert(~isempty(dstInd), ['ID: ', num2str(dstInd), ' is not found']);
            obj.networkArchitecture(srcInd, dstInd) = params;
            srcInds = obj.blocksInds(srcInd,1):obj.blocksInds(srcInd,2);
            destInds = obj.blocksInds(dstInd, 1):obj.blocksInds(dstInd, 2);
            switch params.type
                case 'rand'
                    W = sprand(length(destInds), length(srcInds),  params.connectivity);
                    W(W ~= 0) = W(W ~= 0) - 0.5;
                case 'randn'
                    W = sprandn(length(destInds), length(srcInds), params.connectivity);
            end
            obj.W(destInds, srcInds) = W;
        end
        %------------------------------------------------------------------------------------
        function visualize(obj)
            conMat = obj.getConnectivityMatrix();
            graph = biograph(conMat);
            graph.ArrowSize = 5;

            for k = length(obj.layers):-1:1
                switch(obj.layers{k}.layerType)
                    case 'input'
                        graph.Nodes(k).Color = [0.7 0.7 0.5];
                        graph.Nodes(k).shape = 'parallelogram';
                        graph.Nodes(k).Label = [num2str(obj.layers{k}.id), ':',obj.layers{k}.layerType];
                    case 'output'
                        graph.Nodes(obj.layers{k}.id).Color = [0.7 0.7 0.5];
                        graph.Nodes(k).shape = 'parallelogram';
                        graph.Nodes(k).Label = [num2str(obj.layers{k}.id), ':',obj.layers{k}.layerType];
                    case 'reservoir'
                        graph.Nodes(k).shape = 'circle';
                        graph.Nodes(k).LineWidth = 10 * obj.layers{k}.params.connectivity;
                        numIntStates = obj.blocksInds(end,end) -  obj.blocksInds(2,end); % *needs take into account deleted layers
                        graph.Nodes(k).Size = round(80 * [obj.layers{k}.numNodes; obj.layers{k}.numNodes]/numIntStates) + 20;
                        graph.Nodes(k).Label = [num2str(obj.layers{k}.id), ':',obj.layers{k}.layerType];
                    case 'bias'
                        graph.Nodes(k).shape = 'box';
                        graph.Nodes(k).Size = [5; 5];
                        graph.Nodes(k).Label = [num2str(1)];
%                         pos = graph.Nodes(obj.idToInd(obj.getId('input'))).Position;
%                         graph.Nodes(k).Position = [pos(1), pos(2) + 10];
                end  
                
            end
            
            for k = 1:length(graph.Edges)
                if(isempty(strfind(graph.Edges(k).ID, ['Node ', num2str(obj.idToInd(obj.getId('bias')))])))
                    graph.Edges(k).LineWidth = 10 * graph.Edges(k).Weight;
                else
                    graph.Edges(k).LineWidth = 0.1;
                end
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
            ind = obj.idToInd(obj.getId('output'));
            y = obj.x(obj.blocksInds(ind, 1):obj.blocksInds(ind, 2));
        end
        %-------------------------------------------------------------------------------------
        function setInput(obj, u)
            ind = obj.idToInd(obj.getId('input'));
            obj.x(obj.blocksInds(ind, 1):obj.blocksInds(ind, 2)) = u;
        end
        %-------------------------------------------------------------------------------------
        function x = activate(obj, x)
            for k = 1:length(obj.layers)
            	ind = obj.idToInd(obj.layers{k}.id); %% could be speeded up
                inds = obj.blocksInds(ind,1):obj.blocksInds(ind,2);
                x(inds) = obj.layers{k}.actFunc(x(inds)); %this part a bit slows down
            end
        end
        %-------------------------------------------------------------------------------------
        function setWeights(obj, id1, id2, W)
            ind1 = obj.idToInd(id1);
            ind2 = obj.idToInd(id2);
            inds1 = obj.blocksInds(ind1,1):obj.blocksInds(ind1,2);
            inds2 = obj.blocksInds(ind2,1):obj.blocksInds(ind2,2);
            obj.W(inds1, inds2) = W;
        end
        %-------------------------------------------------------------------------------------
        function W = getWeights(obj, id1, id2)
            ind1 = obj.idToInd(id1);
            ind2 = obj.idToInd(id2);
            inds1 = obj.blocksInds(ind1,1):obj.blocksInds(ind1,2);
            inds2 = obj.blocksInds(ind2,1):obj.blocksInds(ind2,2);
            W = obj.W(inds1, inds2); 
        end
        %-------------------------------------------------------------------------------------
        function x = forward(obj, u) % Calculates next step of the system
            obj.setInput(u);
            obj.x = obj.leakage .* obj.activate(obj.W * obj.x) + (1 - obj.leakage) .* obj.x;
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
            obj.forward(input(:,1));
            y = zeros(genLenght, 1);
            y(1) = obj.getOutput();
            for k = 2 : genLenght
                obj.forward(y(k - 1));
                y(k) = obj.getOutput();
            end
        end
       %------------------------------------------------------------------------------------- 
       function idList = getId(obj, name)
            idList = [];
            for k = 1:length(obj.layers)
                if(strcmp(obj.layers{k}.layerType, name))
                    idList = [idList, obj.layers{k}.id];
                end
            end
       end
       %------------------------------------------------------------------------------------- 
       function ind = idToInd(obj, id)
            ind = find(obj.IDs == id); % could be speeded up
       end
       %------------------------------------------------------------------------------------- 
       function id = indToId(obj, ind)
            id = obj.IDs(ind);
       end       
       %------------------------------------------------------------------------------------- 
       function idConTo = getConnectedTo(obj, id)
             conMat = obj.getConnectivityMatrix();
             indConTo = conMat(:,obj.idToInd(id)) ~= 0;
             idConTo = obj.indToId(indConTo);
       end
       %------------------------------------------------------------------------------------- 
       function stateInds = getInds(obj, ids)
            stateInds = [];
            for k = 1:length(ids)
                indLayer = obj.idToInd(ids(k));
                stateInds = [stateInds, obj.blocksInds(indLayer,1):obj.blocksInds(indLayer,2)];
            end
       end
    end
end