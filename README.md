# deep_learning_tasks
Collection of all deep learning ipynb tasks

## Assignment 5 workflow:-
1. MNIST dataset
2. Using tensorflow to play with the dataset by first incrementally changing the number of hidden layers from 1 to 3, then the activation function and finally the dropout hyperparameter.
3. Plot the graph for loss vs epoch and accuracy(train, validation accuracy) vs epoch for all the above cases. Point out the logic in the report.
4. With the best set hyperparameter from above run vary the Adam Optimizer learning rate [0.01, 0.001, 0.005, 0.0001, 0.0005]. Print the time to achieve the best validation accuracy (as reported
   before from all run) for all these five run .
5. Create five image(size 28*28) containing a digit of your own handwriting and test whether your trained classifier is able to predict it or not.


## Assignment 6 workflow:-
1. Train a Convolutional neural network with max pooling and a fully connected layer at the top, to classify the flower images(Kaggle dataset). And run the network by changing the hyper-parameters.
2. Plot the graph for the loss vs epoch and accuracy(train, test set) vs epoch for all the above cases. Also, plot the accuracy for all experimentation in a bar graph along with the confusion matrix and F1 score.
3. For the best model on the MINST dataset in Assignment 5, train a model with MNIST data using the best set of parameters obtained in Question 4. Compare the test accuracy and also the self-      
   created images.


## Assignment 7 workflow:-
1. Download and preprocess the sentiment analysis dataset from https://www.kaggle.com/snap/amazon-fine-food-reviews. Download the Glove word vectors from http://nlp.stanford.edu/data/glove.6B.zip 
   and extract the 100-dimensional file (glove.6B.100d.txt) from the zipped folder.
2. Preprocess the review dataset by considering the column “review score” >3 as positive reviews and others as negative reviews. For training on local machine considers 5000 positive and 
   negative reviews each for the training dataset.
   Consider 2000 reviews for the test dataset and validation dataset each. Strip the length of each review sentence (number of words) according to your computation availability.
3. Playing with the hyperparameters.
4. Compare the number of parameters, training and inference computation time, Training Loss graph (preferably in a single graph), accuracy.
5. For the best model try the Hindi movie review dataset https://www.kaggle.com/disisbig/hindi-movie-reviews-dataset (use self trainable embedding layer or any other Hindi Word2Vec     
   representation).
