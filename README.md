# Sentiment-Network
Built a Neural Network from scratch for performing the sentiment analysis of the reviews. Open notebook to see the detailed analysis with plots. 

## Features:
- 3 layer neural network
- 10 Nodes in the hidden layer
- Non-linearity was removed from the hidden layer (ie. No Activation Function was used)
- Achieved an accuracy of 87% on the test set
- Achieved an speed of 4400 reviews/sec on my system
- Analysis is performed for reducing the noise by finding the polarity of each word. Words havind low predictive power were removed from the vocab. This  helped in reducing the vocab size and thus increasing the speed of training and accuracy. 
- Plots have been used to show the polarity of words
- Network was able to identify words with similar sentiments (shown using tf-idf)


This project is performed in a progressive manner. First a simple approach is tried and then changes are made to improve the performance.

## Running the project:
Activate a python3 conda environment. In this environment, you'll need to have installed:
- numpy
- jupyter notebook 
- matplotlib 
- scikit-learn and 
- bokeh

Change directories into the downloaded folder.

Start up your Jupyter notebook server.

Open SentimentNetwork.ipynb.

## Some Takeaways:
1)During this project, I noticed that initializing weights is an important step, and one of the problems why your neural network might not work is wrong intialization of weights. 
Why/How initialization of the weights matter ? 
Let us see how should we NOT initialise the weights...

=> What happens when the weights of ALL (there is difference betseen some layers initialised to zero and all layers initialized to zero) the layers are initialised to zero?

In backpropogation all the weight updates will be killed to 0 hence we will be stuck !

=> What if we initialize with small weights ? => Will work in small networks and in deep networks again the input will be killed after it will propogate through some layers and also during backpropogation the gradients will get killed and hence no updates!

=> What if we initialize by large number ? => That the neurons will get saturated, and gradients will get killed during backpropogation.

So? Use Xavier's Initialisation (w = np.random.randn(fan_in, fan_out)/np.sqrt(fan_in)), it's derivation assumes linear activations

For Relu Non Linear activations we need to add a factor of 2; (w = np.random.randn(fan_in, fan_out)/np.sqrt(fan_in/2))

or Use Batch Normalization ! 

For more information on this watch this: https://www.youtube.com/watch?v=GUtlrDbHhJM&list=PLlJy-eBtNFt6EuMxFYRiNRS07MCWN5UIA&index=5 

2)Activation function should be zero-centered, should not kill gradients ie should not saturate and computationally efficient. Here I removed the non-linearity from the hidden layer, ie I removed the activation function sigmoid because the sigmoid is not zero centered. Why ? What's the problem ? Here, The inputs to the network are always positive. So the gradients for the weight at a particular iteration of backpropogation will either be all negative or all positive. So we are restricted in 2 directions where we can go and hence we are restricted. This will cause problem in learning. This is also the reason you always want a zero-mean data. We want zero centered things in the input. Zero centered things throughout! Instead of removing the non-linearity I could have also used used another activation function like tanh(x) which is zero centered! but it also has problems such as kill gradient when saturated. 

For more information on this watch this: https://www.youtube.com/watch?v=GUtlrDbHhJM&list=PLlJy-eBtNFt6EuMxFYRiNRS07MCWN5UIA&index=5 

3)***tf-idf*** can be used to reduce the dimension of the vectors into the the specified dimension. And the results can be plotted(which would have been difficult in high dimension) to compare the vectors.

4)Improving the speed of training is important. Why ? Obviously this will help us train our network faster and if the training data is very large, it is more beneficial to cover the data faster as we will know more about the varaince in our data set which will help in increasing the accuracy. Also this will help us reduce our idle time. :P

5)We should analyse the input, what we are feeding to the network => this will help us identify the noise in our input and thus improve our prediction accuracy and training speed !


## Credits:
This project was performed for learning purpose! Thank You Andrew Trask (@iamtrask) :)  
