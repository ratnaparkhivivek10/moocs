function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


predicted = sigmoid(X*theta);
actual = y;

loss = ((actual.*log(predicted)) + ((1-actual).*log(1-predicted)));
reg = (lambda/(2*m))*sum(theta(2:length(theta)).^2);
cost = (-1/m)*sum(loss)+reg;
J = cost;

error = predicted - actual;
% grad = (1/m)*(X' * error)+(lambda/m)*theta(2:length(theta));
% Submission failed: operator +: nonconformant arguments (op1 is 3x1, op2 is 2x1)
theta(1) = 0
grad = (1/m)*(X' * error)+(lambda/m)*theta;

J
grad



% =============================================================

end
