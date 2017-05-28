function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% X1=5000*401	y=5000*1	Theta1=25*401	Theta2=10*26	y1 = 5000*10



%for i = 1:m
%	if y(i) == 1 
%		y1(i,:) = [1,0,0,0,0,0,0,0,0,0];
%	elseif y(i) == 2 
%		y1(i,:) = [0,1,0,0,0,0,0,0,0,0];
%	elseif y(i) == 3 
%		y1(i,:) = [0,0,1,0,0,0,0,0,0,0];
%	elseif y(i) == 4 
%		y1(i,:) = [0,0,0,1,0,0,0,0,0,0];
%	elseif y(i) == 5 
%		y1(i,:) = [0,0,0,0,1,0,0,0,0,0];
%	elseif y(i) == 6 
%		y1(i,:) = [0,0,0,0,0,1,0,0,0,0];
%	elseif y(i) == 7 
%		y1(i,:) = [0,0,0,0,0,0,1,0,0,0];
%	elseif y(i) == 8 
%		y1(i,:) = [0,0,0,0,0,0,0,1,0,0];
%	elseif y(i) == 9 
%		y1(i,:) = [0,0,0,0,0,0,0,0,1,0]	;	
%	elseif y(i) == 10
%		y1(i,:) = [0,0,0,0,0,0,0,0,0,1];
%	end
%end

%A more complicated way of doing what I did above:
yd = eye(num_labels);
y1 = yd(y,:);

% a1=5000*401	y=5000*1	Theta1=25*401	Theta2=10*26	y1 = 5000*10

%size(y1)
	
a1 = [ones(m,1),X];
z2 = a1*Theta1';	%5000*25
a2 = sigmoid(z2);	%5000*25 into 5000x26
a2 =[ones(m,1),a2];
z3 = a2*Theta2';	%5000*10
a3 = sigmoid(z3); 	%5000*10

%print 'a3 is'
%a3

J = (-1/m) * sum(sum( y1.*log(a3) + (1.-y1).*log(1-(a3)) ))	; 

J = J + (lambda/(2*m))*( sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)) );


ddelta_2 = 0;
ddelta_1 = 0;

delta_3 = a3 .- y1; %5000*10
delta_2 = delta_3*Theta2(:,2:end) .* sigmoidGradient(z2);	% 5000*25

ddelta_2 = ddelta_2 + delta_3' * a2; 	%
ddelta_1 = ddelta_1 + delta_2' * a1;	% 25x401 

Theta1_grad = 1/m .* ddelta_1 + (lambda/m) *[zeros(hidden_layer_size,1), Theta1(:,2:end)];
Theta2_grad = 1/m .* ddelta_2 + (lambda/m) *[zeros(num_labels,1), Theta2(:,2:end)];



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
