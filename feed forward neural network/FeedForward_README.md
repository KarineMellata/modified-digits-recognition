FeedForward Neural Network

Make sure train_x, train_y, test_x, and all the matlab files are in the same directory.

Run the MLP.m file and it will utilize the other matlab files and run the code. The variable final will be the output given the inputs X and y_actual in the 'check' method in the MLP.m file. The learning rate alpha and lambda can be modified in the code before it is passed into the backpropagation method. 

Finally, depending on whether you want to train on the entire training set or just on the "train set" (i.e. the first 30000 entries), then a modification of the line

   i = randi([1 50000], 1, 1) 

is necessary. For the entire training set, the number should remain 50000, however, should you wish to only train on 60% of the entire training set, that number needs to be modified to 30000, and these two lines

X_head = X(i,:);
y_head = y_actual(i,:);

need X and y_actual to be replaced with X_train and y_train respectively. 

