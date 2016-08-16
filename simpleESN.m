% load the data
trainLen = 2000;
testLen = 2000;
initLen = 0;

%data = load('MackeyGlass_t17.txt');
% plot some of it
% figure(10);
% plot(data(1:1000));
% title('A sample of data');

% generate the ESN reservoir
inSize = 1; outSize = 1;
resSize = 50;
a = .5; % leaking rate
connectivity = 0.2;
rand( 'seed', double(cputime) );
Win = (rand(resSize,1+inSize)-0.5) .* 1;
%W = rand(resSize,resSize)-0.5;
W = sprand(resSize, resSize, connectivity);
W(W ~= 0) = W(W ~= 0) - 0.5;
% Option 1 - direct scaling (quick&dirty, reservoir-specific):
% W = W .* 0.13;
% Option 2 - normalizing and setting spectral radius (correct, slower):
disp 'Computing spectral radius...';
opt.disp = 0;
rhoW = abs(eigs(W,1,'LM',opt));
disp 'done.'
W = W .* ( 1.25 /rhoW);

% allocated memory for the design (collected states) matrix
X = zeros(1+inSize+resSize,trainLen-initLen);
% set the corresponding target matrix directly
Yt = data(initLen+2:trainLen+1)';

% run the reservoir with the data and collect X
x = zeros(resSize,1);
for t = 1:trainLen
	u = data(t);
	x = (1-a)*x + a*tanh( Win*[1;u] + W*x );
	if t > initLen
		X(:,t-initLen) = [1;u;x];
	end
end

% train the output
%reg = 1e-3;  % regularization coefficient
%X_T = X';
%Wout = Yt * X_T * inv(X*X_T + reg*eye(1+inSize+resSize));
%Wout = Yt*pinv(X);
Xinv = pseudoinverse(X, [],'lsqr');
Wout = Yt*Xinv;
%figure, plot(Wout * X)

% run the trained ESN in a generative mode. no need to initialize here, 
% because x is initialized with training data and we continue from there.
Y = zeros(outSize,testLen);
u = data(trainLen+1);
for t = 1:testLen 
	x = (1-a)*x + a*tanh( Win*[1;u] + W*x );
	y = Wout*[1;u;x];
	Y(:,t) = y;
	% generative mode:
	u = 1.0 * y;
	% this would be a predictive mode:
	%u = data(trainLen+t+1);
end

errorLen = 50;
mse = sum((data(trainLen+2:trainLen+errorLen+1)'-Y(1,1:errorLen)).^2)./errorLen;
disp( ['MSE = ', num2str( mse )] );

% plot some signals
figure(1);
plot( data(trainLen+2:trainLen+testLen+1), 'color', [0,0.75,0] );
hold on;
plot( Y', 'b' );
hold off;
axis tight;
title('Target and generated signals y(n) starting at n=0');
legend('Target signal', 'Free-running predicted signal');
% 
% figure(2);
% plot( X(1:20,1:200)' );
% title('Some reservoir activations x(n)');
% 
% figure(3);
% bar( Wout' )
% title('Output weights W^{out}');
