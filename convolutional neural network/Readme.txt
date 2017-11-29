COMP 551 Project 3
Convolutional Neural Network

Author
Xingwei Cao
260572128

Run command
python train.py

Output
.csv file outputs the result of training.

Functions
	
	-iter_loadtxt
		Input	: name of the file
		Output 	: np array of csv file

	-one_hot
		Input	: sparse list of labels
		Output 	: one-hot encoded list of labels

	-next_batch
		Input	: size of batch, x and y 
		Output 	: batch of specified size batch[0] is x and batch[1] is y

	-main
		Input 	: None
		Output 	: .csv file of training CNN with 5 convolutional layers and 2 fully connected layers