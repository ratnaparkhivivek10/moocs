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
a_1 = [ones(m, 1) X];
z_2 = a_1*Theta1';
a_2 = sigmoid(z_2);

a_2 = [ones(m, 1) a_2];
z_3 = a_2*Theta2';
a_3 = sigmoid(z_3);

y = repmat([1:num_labels], m, 1) == repmat(y, 1, num_labels);
J = (-1 / m) * sum(sum(y.*log(a_3) + (1 - y).*log(1 - a_3)));

%Remove bias column from Theta1 and Theta2 for reg cost calculation.
reg_Theta1 =  Theta1(:,2:end);
reg_Theta2 =  Theta2(:,2:end);

error = (lambda/(2*m)) * (sum(sum(reg_Theta1.^2)) + sum(sum(reg_Theta2.^2)));

J = J + error;

%Implement Backpropogation
%WIP
%Iterate over every (1*401) size data point.
%Theta1 = (25*401)
%Theta2 = (10*26)

%One-hot encode the values
y_new = zeros(num_labels, m); % 10*5000
for i=1:m
  y_new(y(i),i)=1;
end

for t=1:m
	% Forward propogataion
	row = a_1(t,:); %(1*400)
	row = row'; %(401*1)
	a1 = Theta1*row; %(25*401) * (401*1) = (25*1)
	z1 = sigmoid(a1);
	
	z1 = [1 ; z1]; % adding a bias (26*1)
	
	a2 = Theta2*z1; %(10*26) * (26*1) = (10*1)
	z2 = sigmoid(a2); % Output layer activation (10*1)
	
	% Backpropogation
	
	delta_3 = z2 - y_new(:,t); % (10*1)
	a1 = [1; a1]; % bias (26*1)
	delta_2 = (Theta2' * delta_3) .* sigmoidGradient(a1); % ((26*10)*(10*1))=(26*1)
	delta_2 = delta_2(2:end);
	
	Theta2_grad = Theta2_grad + delta_3 * z1'; % (10*1)*(1*26)
	Theta1_grad = Theta1_grad + delta_2 * row'; % (25*1)*(1*401)
	
end

Theta2_grad = (1/m) * Theta2_grad; % (10*26)
Theta1_grad = (1/m) * Theta1_grad; % (25*401)


Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + ((lambda/m) * Theta1(:, 2:end));
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + ((lambda/m) * Theta2(:, 2:end));


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
