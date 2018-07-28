"""
A template for the first assignment.
See https://github.com/text-machine-lab/uml_nlp_class/tree/master/hw1/README.md for details.
"""

from datetime import datetime
from collections import Counter

import numpy as np
import torch
import torch.cuda
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

import random

TEXT8_FILENAME = './text8.txt'
TEST_WORDS_FILENAME = './test_words.txt'


def load_data(filename):
    with open(filename, 'r') as f:
        data = f.read()

    tokens = data.split()

    return tokens


def load_test_words(filename):
    with open(filename, 'r') as f:
        words = f.read().split()

    return words


def build_dataset(data, vocab_size=50000):
    """
    Build the dataset (convert words to indices) and return it, along with mapping word <-> id
    :param data: Aa list of tokens
    :param vocab_size: Maximum size of the vocabulary
    :return: A list of tokens ids, a dictionary token -> id, a dictinary id -> token
    """

    # we will replace non-frequent tokens with the `unknown` token
    unk_token = '<UNK>'

    # calc frequencies of the tokens in our data
    tokens_counts = Counter(data)
    most_common_tokens = tokens_counts.most_common(vocab_size)

    # create a token => id mapping
    token2id = {unk_token: 0}
    for token, counts in most_common_tokens:
        token2id[token] = len(token2id)

    # create a reverse mapping from ids to tokens
    id2token = {i: t for t, i in token2id.items()}

    # convert data to tokens ids
    nb_unks = 0
    data_tokens_ids = []
    for token in data:
        if token in token2id:
            idx = token2id[token]
        else:
            idx = token2id[unk_token]
            nb_unks += 1

        data_tokens_ids.append(idx)

    print('Vocab size:', len(token2id))
    print('Unknown tokens:', nb_unks)

    return data_tokens_ids, token2id, id2token


class SkipGramDataset(torch.utils.data.Dataset):
    def __init__(self, data, skip_window=1):
        super().__init__()
        self.data = data
        self.skip_window = skip_window

    def __getitem__(self, index):
        """
        Return a training sample from the dataset
        :param index: index of the sample
        :return: a tuple (target_word index, context word index)
        """
        target_word_index = self.data[index + self.skip_window]
        random_index = random.randint(0 , 2*self.skip_window -1)
        if random_index >= self.skip_window:
            context_word_index = self.data[index + (random_index - self.skip_window  + 1)]
        else:
            context_word_index = self.data[index + (random_index - self.skip_window)]
        #print ('getitem:', target_word_index, context_word_index)
            
        return (target_word_index, context_word_index)
        
        #raise NotImplementedError('Implement the __getitem__ method of the dataset')

    def __len__(self):
        """
        Get the length of the dataset
        :return: length of the dataset
        """        
        length = len(self.data) - 2* self.skip_window
        #print ('length', length)
        return length
        #raise NotImplementedError('Implement the __len__ method of the dataset')


class SkipGramModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.fc1 = nn.Embedding (vocab_size, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, vocab_size)
        self.out = []
    

    def forward(self, inputs):
        """
        Perform the forward pass of the model
        :param inputs: A tensor of size (batch size, ) of indices of the target words
        :return: A tensot of the shape of (batch size, vocab size) of unnormalized probabilities of the words being the context words
        """
        #print (len(inputs))
        out = self.fc1(inputs)
        out = self.fc2(out)
        self.out = out
        return out
        #raise NotImplementedError('Implement the forward method of the model')

    def find_closest(self, inputs, nb_closest=5):
        """
        Find the closest word using cosine similarity
        :param inputs: A tensor of size (batch size, ) of indices of the target words
        :param nb_closest: The number of closest word to return
        :return: A tensor of size (bach size x nb_closest) containing indices of the closest words
        """
        result = np.zeros((2,6))
        output = self.out.clone()
        op = output.transpose(0,1)
        ip = op[inputs]
        input_prob = ip.transpose(0,1)        
        cos = nn.CosineSimilarity(0)
      
        for in_col, token_id in enumerate(inputs.data.cpu().numpy()):
          cosines = None
          
          for o_col, prob in enumerate(output.data.cpu().numpy()):
           # if token_id == o_col:
           #   continue
            c = cos (input_prob.data[:,in_col], output.data[:,o_col])
            if (cosines is not None):
              cosines = torch.cat([cosines,c])
            else :
              cosines = c
              
          #print(cosines.size())
          res = torch.topk(cosines, nb_closest+1)
          #print (type(res[1]))
          res = Variable (res[1])
          #print (type (res))
          res = res.data.cpu().numpy()
          result = np.concatenate((result,[res]),axis=0)
        
        result = np.delete(result, 0, 0)
        result = np.delete(result , 0, 0)
        return (cuda(Variable(torch.from_numpy(result))))
          
        #raise NotImplementedError('Implement the find_closest method of the model')


def cuda(obj):
    if torch.cuda.is_available():
        obj = obj.cuda()

    return obj


def main():
    vocab_size = 50000
    embedding_dim = 128
    skip_window = 1

    batch_size = 1024

    data = load_data(TEXT8_FILENAME)
    print('Data:', len(data))
    print(data[:10])

    data_tokens_ids, token2id, id2token = build_dataset(data, vocab_size=vocab_size)
    print('Dataset:', len(data_tokens_ids))
    print(data_tokens_ids[:10])

    # load words to test embeddings
    test_words = load_test_words(TEST_WORDS_FILENAME)
    
    dataset = SkipGramDataset(data_tokens_ids, skip_window=skip_window)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    #print ('token to id :', len(token2id))
    model = SkipGramModel(vocab_size=len(token2id), embedding_dim=embedding_dim)
    model = cuda(model)
    
    # cross-entropy loss is used as the models outputs unnormalized probability distribution over the vocabulary
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adagrad(model.parameters(), lr=0.1)

    print('Starting training...')
    nb_iterations = 0
    iteration_step = 1000
    max_iterations = 20000

    losses = []
    time_training_started = datetime.now()
    tick = datetime.now()
    while True:
        for i, (inputs, targets) in enumerate(data_loader):
            optimizer.zero_grad()
            #print (i, inputs, targets)

            inputs_var = cuda(Variable(inputs))
            targets_var = cuda(Variable(targets))
            #print (inputs_var, targets_var)
            
            outputs = model(inputs_var)
            #print (outputs[:10])
  
            loss = criterion(outputs, targets_var)
            loss.backward()
            optimizer.step()

            losses.append(float(loss))
            nb_iterations += 1

            if nb_iterations % iteration_step == 0:
                tock = datetime.now()
                time_for_iterations = tock - tick
                time_elapsed = tock - time_training_started

                loss = np.mean(losses)
                losses = []

                log_str = 'Iteration: {}, elapsed time: {} [{} sec/{} iters], loss: {}'.format(
                    nb_iterations, time_elapsed, time_for_iterations.seconds, iteration_step, loss
                )
                print(log_str)

                # find closest word and print them
                test_samples = cuda(Variable(torch.from_numpy(np.array([token2id[t] for t in test_words]))))
                random_idx = torch.from_numpy(np.random.choice(batch_size, 3))
                #print (random_idx)
                random_words_from_batch = cuda(Variable(inputs[random_idx]))
                #print (inputs[random_idx])
                test_samples = torch.cat([test_samples, random_words_from_batch])
                #print (test_samples.size())
                closest = model.find_closest(test_samples)
                #print (closest)
                closest = closest.data.cpu().numpy()
                

                # print some random samples
                for i, token_id in enumerate(test_samples.data.cpu().numpy()):
                    target_token = id2token[token_id]
                    #print (i)
                    closest_tokens = [id2token[i] for i in closest[i]]
                    print('Closest to {}: {}'.format(target_token, ', '.join(closest_tokens)))
                print()

                tick = datetime.now()

            if nb_iterations > max_iterations:
                return


if __name__ == '__main__':
    main()