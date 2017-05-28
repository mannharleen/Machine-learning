function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%theta:2*1	X:m*2 y:m*1
%theta:9*1	X:m*9 y:m*1

%J = sum((h-y).^2)/2/m + lambda/2/m*(sum(theta.^2)-theta(1)^2);
J = 1/(2*m) * sum((X * theta - y).^2) + lambda/(2*m) .* sum(theta(2:end,1).^2);
a = 1/m .* sum((X * theta - y).*X(1:m,1)) ;
%fprintf('size a:' );
%size(a)
%b = 1/m .* sum(sum((X * theta - y).*X(1:m,2:end))) .+ lambda.*theta(2:end,1)./m;
% 8*1				m*1					m*8
b = 1/m .* (X(1:m,2:end)' * (X * theta - y)) .+ lambda.*theta(2:end,1)./m;
%fprintf('size b:' );
%size(b)
%size(a)
%size(b)
grad = [a;b];












% =========================================================================

grad = grad(:);

end
