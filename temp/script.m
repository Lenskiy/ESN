combinedMeanSSE(:, 1) = mean(rnnsse01);
combinedMeanSSE(:, 2) = mean(rnnsse001(1:9,:)');
combinedMeanSSE(:, 3) = mean(sse001{1});
combinedMeanSSE(:, 4) = mean(sse001{2});
combinedMeanSSE(:, 5) = mean(sse001{3});
combinedMeanSSE(:, 6) = mean(sse001{4});
combinedMeanSSE(:, 7) = mean(sse001{5});
combinedMeanSSE(:, 8) = mean(sse005leak{1});
combinedMeanSSE(:, 9) = mean(sse005leak{2});
combinedMeanSSE(:, 10) = mean(sse005leak{3});
combinedMeanSSE(:, 11) = mean(sse005leak{4});
%combinedMeanSSE(:, 12) = mean(sse005leak{5});

combinedSTDSSE(:, 1) = std(rnnsse01);
combinedSTDSSE(:, 2) = std(rnnsse001(1:9,:)');
combinedSTDSSE(:, 3) = std(sse001{1});
combinedSTDSSE(:, 4) = std(sse001{2});
combinedSTDSSE(:, 5) = std(sse001{3});
combinedSTDSSE(:, 6) = std(sse001{4});
combinedSTDSSE(:, 7) = std(sse001{5});
combinedSTDSSE(:, 8) = std(sse005leak{1});
combinedSTDSSE(:, 9) = std(sse005leak{2});
combinedSTDSSE(:, 10) = std(sse005leak{3});
combinedSTDSSE(:, 11) = std(sse005leak{4});
%combinedSTDSSE(:, 12) = std(sse005leak{5});


plotPredictionAccuracy(combinedMeanSSE, combinedSTDSSE, [20,63,147,263,411,591,803,1047,1323],...
    {'BP RNN, eta = 0.1',...
     'BP RNN, eta = 0.01', ...
     'ESN, con = 0.01, rad = 0.25', ...
     'ESN, con = 0.01, rad = 0.5', ...
     'ESN, con = 0.01, rad = 0.75',...
     'ESN, con = 0.01, rad = 1.0',...
     'ESN, con = 0.01, rad = 1.25', ...
     'ESN, con = 0.05, rad = 0.25',... 
     'ESN, con = 0.05, rad = 0.5', ...
     'ESN, con = 0.05, rad = 0.75',...
     'ESN, con = 0.05, rad = 1.0',...
     'ESN, con = 0.05, rad = 1.25'}, 1, struct('linewidth', 2));
