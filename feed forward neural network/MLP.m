function output = MLP
X = csvread('train_x.csv'); %matrix of 50000 by 4096
y = csvread('train_y.csv'); %matrix of 50000 by 1
indices = find(abs(X)<215);
X(indices) = [0];
classes =  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 24, 25, 27, 28, 30, 32, 35, 36, 40, 42, 45, 48, 49, 54, 56, 63, 64, 72, 81];
X = X./255;

y_actual = zeros(50000, 40);   
for i = (1:50000)
    index = find(classes == y(i));
    y_actual(i, index) = [1];
end

W1_2 = randi([-10 10], 4096, 40)*0.005; 
W2_3 = randi([-10 10], 40, 40) * 0.005;

alpha=0.0003;
lambda=0.0001;
X_train = X(1:30000, :);
X_val1 = X(30001:35000, :);
X_val2 = X(35001:40000, :);
X_val3 = X(40001:45000, :);
X_val4 = X(45001:50000, :);
y_train = y_actual(1:30000, :);
y_val1 = y_actual(30001:35000, :);
y_val2 = y_actual(35001:40000, :);
y_val3 = y_actual(40001:45000, :);
y_val4 = y_actual(45001:50000, :);
for j = (1:35000)
    i = randi([1 50000], 1, 1);
    X_head = X(i,:);
    y_head = y_actual(i,:);
    [W1_2, W2_3] = backprop(X_head, y_head, W1_2, W2_3, alpha, lambda);
    if (mod(j, 25) == 0)
        temp1 = sigmoid(X_val1*W1_2);
        output = softmax(temp1*W2_3);
            [correct, incorrect, final] = check(X_val2, W1_2, W2_3, y_val2, classes, output); %modify X and y_actual values here if evaluation is to be done on a validation set
    end
end
temp1 = sigmoid(X*W1_2);
output = softmax(temp1*W2_3);


end

    
