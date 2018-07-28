# Working-with-word-vectors
In the first part I have used the fastText library to train your very own word vectors on the Text8 data. In the second part I have implemented and trained a Word2Vec skip-gram modelon the same data.

Part one: (h1_prob1.py)
For implementing part one I have used the sample code provided which uses the inbuilt pyfasttest library and 
added kmeans clustering to that for every selected word.

Part two: (h1_prob2.py)
I have completed this part as follows:
A: For SkipGramDataset class, I have implemented:
	1. the _len_ method by subtracting the skip window size from front and back; 
	2. the _getitem_ method by selecting a random number in size 0 to 2*skip_window - 2 and selected a context_word from the dataset around the target_word after
	skipping the target_word based on random number generated.
	
	The output can be seen in h1_part1_op.txt
	
B: For SkipGramModel
	1. _init()_ method is declaring the layers and also saving the output at current layer for use in find_closest()
	2. forward() method is just calling the defined layers to get the probability distribution
	3. find_closest() method uses the output saved for the current layer and gets the probability distribution for the input words
					it then calls the nn.CosineSimilarity() method on the every input word for every column of the output (i.e., vocab words) and 
					appends it to an array. it then calculates the top 6 elements using topk function 
					(top 6 because the output has the input words copy as well for now I have not deleted that copy).
					All these top 6 words are then saved in an array and returned to the caller function after wrapping it in cuda Variable
					
	The output can be seen in h1_part2_op.txt
