import warnings
import numpy as np
from itertools import chain
from functools import reduce
from data_utils import load_task, vectorize_data

# Ignore warning that cross_validation is deprecated
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category = DeprecationWarning)
    from sklearn import cross_validation

class Data:
    def __init__(self, data_dir, memory_size = 50, num_tasks = 20):
        # Read data
        self.train_set, self.test_set = zip(*[load_task(data_dir, i) for i in range(1, num_tasks + 1)])
        self.data = flatten_sublists(self.train_set + self.test_set)

        # If given memory size is too large, reset it as necessary
        max_story_size = max([len(story) for story, _, _ in self.data])
        self.memory_size = min(memory_size, max_story_size)
        self.word_to_index = None

    def read_data(self):
        # Build dictionary of words
        dictionary = set([])
        for story, query, answer in self.data:
            story_words = flatten_sublists(story)
            dictionary.update(story_words + query + answer)

        # Build one-hot indices for each word, adding time words
        self.word_to_index = {word: index for index, word in enumerate(dictionary)}
        for i in range(1, self.memory_size + 1):
            time_word = "time%s" % str(i)
            self.word_to_index[time_word] = time_word

        self.max_sentence_size = max([len(sentence) for sentence in flatten_sublists([story for story, _, _ in self.data])])
        max_query_size = max([len(query) for _, query, _ in self.data])
        if max_query_size > self.max_sentence_size:
            self.max_sentence_size = max_query_size
        self.max_sentence_size += 1 # Add one for time words

        # Read Babi tasks
        datasets = [[] for _ in range(6)] # Stories, queries, and answers, for training and validation

        # Training and validation data
        for task in self.train_set:
            # Convert data to arrays of embeddings
            S, Q, A = vectorize_data(task, self.word_to_index, self.max_sentence_size, self.memory_size)

            # Split into train and validation
            ts, vs, tq, vq, ta, va = cross_validation.train_test_split(S, Q, A, test_size = 0.1)
            for dataset, data in zip(datasets, (ts, tq, ta, vs, vq, va)):
                dataset.append(data)

        datasets = [reduce(lambda X, Y: np.vstack((X, Y)), dataset) for dataset in datasets]

        training, validation = [], []
        for i in range(len(datasets[0])):
            data_point = tuple((np.array(datasets[j][i]) for j in range(3)))
            training.append(data_point)

        for i in range(len(datasets[3])):
            data_point = tuple((np.array(datasets[j][i]) for j in range(3, 6)))
            validation.append(data_point)

        # Test data
        testing = []
        S, Q, A = vectorize_data(flatten_sublists(self.test_set), self.word_to_index, self.max_sentence_size, self.memory_size)
        for i in range(len(S)):
            data_point = (np.array(l[i]) for l in (S, Q, A))
            testing.append(data_point)

        return (training, validation, testing)

    def vocab_size(self):
        if self.word_to_index is None:
            raise Exception("Data hasn't been read yet")

        return len(self.word_to_index) + 1 # Add one for the nil word
    
    def max_sent_size(self):
        if self.word_to_index is None:
            raise Exception("Data hasn't been read yet")

        return self.max_sentence_size

flatten_sublists = lambda l: list(chain.from_iterable(l)) # Flattens list of lists by one level
