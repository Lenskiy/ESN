% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Generate the ESN reservoir
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

rand('seed', 42);

trainLen = 7000;
testLen  = 2000;
initLen  = 100;
%addpath('../T0_chaotic/');
data = mackeyglass(10000);

clear X;

%         Input neurons
inSize  = 1; 
%         Output neurons 
outSize = 1;
%         Reservoir size
resSize = 1000;
%         Leaking rate
leak       = 0.6; 
%         Input weights
Win     = ( randn(resSize, (inSize+1) )) .* 1;
%         Reservoir weights
W       = sprandn(resSize, resSize, 0.1);
%         Spectral radius
radius = 1;

% Noramize the weigth matrix so the highest eigenvalue is equal to radius
opts.tol = 1e-3;
maxVal = max(abs(eigs(W, 1, 'lm', opts)));
W = W/maxVal;
            
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% Run the reservoir with the data and collect X.
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%       Allocated memory for the design (collected states) matrix
X     = zeros((1+inSize) + resSize, trainLen - initLen);

%       Vector of reservoir neuron activations (used for calculation)
x     = zeros(resSize, 1);

%       Update of the reservoir neuron activations
xUpd  = zeros(resSize, 1);

for t = 1:trainLen

    u    = data(t);

    xUpd = tanh( Win * [1;u] + W * x );    
    x    = (1-leak) * x + leak * xUpd;

    if ( t > initLen )
        X(:,t-initLen) = [1;u;x];
    end

end

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% Train the output
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%       Set the corresponding target matrix directly
Yt    = data(initLen+2:trainLen+1)';

%       Regularization coefficient
reg   = 1e-8;  

%       Get X transposed - needed twice therefore it is a little faster
X_T   = X';

%       Yt * pseudo_inverse(X); (linear regression task)
Wout  = Yt * X_T * (X * X_T + reg * eye(1+inSize+resSize))^(-1);

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% Run the trained ESN in a generative mode. no need to initialize here, 
% because x is initialized with training data and we continue from there.
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Y = zeros(outSize,testLen);
u = data(trainLen+1);

for t = 1:testLen 

    xUpd   = tanh( Win*[1;u] + W*x );
    x      = (1-leak)*x + leak*xUpd;

    %        Generative mode:
    u      = Wout*[1;u;x];

    %      This would be a predictive mode:
    %u      = data(trainLen+t+1);

    Y(:,t) = u;

end

errorLen = 500; 
mse = sum((data(trainLen+2:trainLen+errorLen+1)'-Y(1,1:errorLen)).^2)./errorLen; 
disp( ['MSE = ', num2str( mse )] ); 
% plot some signals 
figure(1); 
plot( data(trainLen+2:trainLen+testLen+1), 'color', [0,0.75,0] ); 
hold on; 
plot( Y', 'b' ); 
legend('True', 'Generated')
%hold off; 
axis tight; 
title('Target and generated signals y(n) starting at n=0'); 
legend('Target signal', 'Free-running predicted signal'); 
% 
figure(2); plot( X(3:52,1:400)' ); 
title('First 50 reservoir states x(n)'); 

figure(3); bar( Wout' ) 
title('Output weights W^{out}');