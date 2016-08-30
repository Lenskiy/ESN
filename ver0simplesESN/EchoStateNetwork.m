function EchoStateNetwork()
clc;close all;clear all
% signal
Yd = [[0.5 .* sin((1:500)/10)].^7;[0.5 .* sin((1:500)/10)]];
% The connectivity density of reservoirs. Usually the value range of D is [0.01 1]
D      = 0.4;
% Number of reservoir neurons
N      = 1500;
% Create echo state network
net = newESN(Yd,N,D);
% Train echo state network
net = ESNTrain(net,Yd);
% Simulate echo state network
n = 500;    % length of Yh
Yh = SimESN(net,n);
% Mean squared normalized error
E = mean((Yd(:,1:n)'-Yh').^2);
disp(['The MSE of model: ',num2str(E)])
% Display
figure('Color','w');
subplot(211);plot(Yd(1,:));hold on
plot(Yh(1,:)','g' )
legend('Target','Modeling')
axis([1 size(Yd,2) min(Yd(1,:)) max(Yd(1,:))])
title('Target 1')
subplot(212);plot(Yd(2,:));hold on
plot( Yh(2,:)','g' )
legend('Target', 'Modeling')
axis([1 size(Yd,2) min(Yd(2,:)) max(Yd(2,:))])
title('Target 2')
end

function W = initweights(n1,n2,D)
mask = (randn(n1,n2) < D);
W    = (rand(n1,n2) - 0.5).*mask;
if n1 == n2

    opt.disp = 0;
    rhoW = abs(eigs(W,1,'LM',opt));
    p    = ( 1.25 /rhoW);
    W = p.*W;
    disp(['spectral radius: ',num2str(p)])
end
end

function y = arctansig(x)
y = log(sqrt((x+1)./(1-x)));
end

function net = ESNTrain(net,Yd)
X = zeros(1+size(net.W,1),size(Yd,2)-1);
x = zeros(size(net.W,1),1);
for k = 1:size(Yd,2)-1
    x = feval(net.fPE,(net.W*x+net.WBack*[1;Yd(:,k)])*(1-net.lr)+net.lr*x);
    X(:,k) = [1;x];
end
M    = X(:,2:end);
T    = feval(net.ARCfO,Yd(:,3:end));
net.Wout = T*M'*inv(M*M'+1e-8*eye(1+size(net.W,1)));
end

function Yh = SimESN(net,n)
Yh   = zeros(size(net.WBack,2)-1,n);
Yh   = zeros(size(net.WBack,2)-1,1);
x    = zeros(size(net.W,1),1);
for k = 1:n
    x = feval(net.fPE,(net.W*x+net.WBack*[1;Yh(:,k)])*(1-net.lr)+net.lr*x);
    Yh(:,k+1) = feval(net.fO,net.Wout*[1;x]);
end
Yh   = Yh(:,1:end-1);
end

function net = newESN(Yd,N,D)
% Transfer function of reservoir neurons
net.fPE    = 'tansig';
% Transfer function of readout neurons
net.fO     = 'tansig';
net.ARCfO  = 'arctansig';
% Leaking rate of reservoir network
net.lr = 0.3;
% Number of readout neurons
L      = size(Yd,1);
% Reservoir weights
net.W     = initweights(N,N,D);
% Feedward weights
net.WBack = initweights(N,1+L,D);
end