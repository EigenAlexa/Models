import warnings, os, re
import numpy as np
from itertools import chain
from functools import reduce

# Ignore warning that cross_validation is deprecated
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category = DeprecationWarning)
    from sklearn import cross_validation

class Data:
    def __init__(self, data_dir, memory_size = 50, num_tasks = 20):
        # Read data
        self.train_set, self.test_set = zip(*[BabiUtils.load_task(data_dir, i) for i in range(1, num_tasks + 1)])
        self.data = flatten_sublists(self.train_set + self.test_set)

        # If given memory size is too large, reset it as necessary
        max_story_size = max([len(story) for story, _, _ in self.data])
        self.memory_size = min(memory_size, max_story_size)
        self.word_to_index = {}
        self.max_sentence_size = -1

    def read_data(self):
        # Build dictionary of words
        dictionary = set([])
        for story, query, answer in self.data:
            story_words = flatten_sublists(story)
            dictionary.update(story_words + query + answer)
        dictionary = sorted(dictionary)

        # Build one-hot indices for each word, adding time words
        self.word_to_index = {word: index + 1 for index, word in enumerate(dictionary)} # Add 1 since index zero is reserved
        for i in range(1, self.memory_size + 1):
            time_word = "time%s" % str(i)
            self.word_to_index[time_word] = time_word

        # Get maximum sentence length for sentence vector representation length
        self.max_sentence_size = max([len(sentence) for sentence in flatten_sublists([story for story, _, _ in self.data])])
        max_query_size = max([len(query) for _, query, _ in self.data])
        if max_query_size > self.max_sentence_size:
            self.max_sentence_size = max_query_size
        self.max_sentence_size += 1 # Add one for time words

        # Read Babi tasks
        datasets = [[] for _ in range(6)] # Stories, queries, and answers, for training and validation
        for task in self.train_set: # Training and validation data
            # Convert data to arrays of embeddings
            S, Q, A = BabiUtils.vectorize_data(task, self.word_to_index, self.max_sent_size(), self.memory_size)

            # Split into train and validation
            ts, vs, tq, vq, ta, va = cross_validation.train_test_split(S, Q, A, test_size = 0.1, random_state = None)
            for dataset, data in zip(datasets, (ts, tq, ta, vs, vq, va)):
                dataset.append(data)

        datasets = [reduce(lambda X, Y: np.vstack((X, Y)), dataset) for dataset in datasets]

        training, validation = [], []
        for i in range(len(datasets[0])):
            data_point = tuple(np.array(datasets[j][i]) for j in range(3))
            training.append(data_point)

        for i in range(len(datasets[3])):
            data_point = tuple(np.array(datasets[j][i]) for j in range(3, 6))
            validation.append(data_point)

        # Test data
        testing = []
        S, Q, A = BabiUtils.vectorize_data(flatten_sublists(self.test_set), self.word_to_index, self.max_sentence_size, self.memory_size)
        for i in range(len(S)):
            data_point = tuple(np.array(l[i]) for l in (S, Q, A))
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

    def word2index(self):
        if self.word_to_index is None:
            raise Exception("Data hasn't been read yet")

        return self.word_to_index

flatten_sublists = lambda l: list(chain.from_iterable(l)) # Flattens list of lists by one level

class BabiUtils:
    @staticmethod
    def load_task(data_dir, task_id, only_supporting=False):
        '''Load the nth task. There are 20 tasks in total.

        Returns a tuple containing the training and testing data for the task.
        '''
        assert task_id > 0 and task_id < 21

        files = os.listdir(data_dir)
        files = [os.path.join(data_dir, f) for f in files]
        s = 'qa{}_'.format(task_id)
        train_file = [f for f in files if s in f and 'train' in f][0]
        test_file = [f for f in files if s in f and 'test' in f][0]
        train_data = BabiUtils._get_stories(train_file, only_supporting)
        test_data = BabiUtils._get_stories(test_file, only_supporting)
        return train_data, test_data

    @staticmethod
    def _tokenize(sent):
        '''Return the tokens of a sentence including punctuation.
        >>> BabiUtils._tokenize('Bob dropped the apple. Where is the apple?')
        ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
        '''
        # Ignore warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category = FutureWarning)
            tokens = [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]

        return tokens


    @staticmethod
    def _parse_stories(lines, only_supporting=False):
        '''Parse stories provided in the bAbI tasks format
        If only_supporting is true, only the sentences that support the answer are kept.
        '''
        data = []
        story = []
        for line in lines:
            line = str.lower(line)
            nid, line = line.split(' ', 1)
            nid = int(nid)
            if nid == 1:
                story = []
            if '\t' in line: # question
                q, a, supporting = line.split('\t')
                q = BabiUtils._tokenize(q)
                #a = BabiUtils._tokenize(a)
                # answer is one vocab word even if it's actually multiple words
                a = [a]
                substory = None

                # remove question marks
                if q[-1] == "?":
                    q = q[:-1]

                if only_supporting:
                    # Only select the related substory
                    supporting = map(int, supporting.split())
                    substory = [story[i - 1] for i in supporting]
                else:
                    # Provide all the substories
                    substory = [x for x in story if x]

                data.append((substory, q, a))
                story.append('')
            else: # regular sentence
                # remove periods
                sent = BabiUtils._tokenize(line)
                if sent[-1] == ".":
                    sent = sent[:-1]
                story.append(sent)
        return data


    @staticmethod
    def _get_stories(f, only_supporting=False):
        '''Given a file name, read the file, retrieve the stories, and then convert the sentences into a single story.
        If max_length is supplied, any stories longer than max_length tokens will be discarded.
        '''
        with open(f) as f:
            return BabiUtils._parse_stories(f.readlines(), only_supporting=only_supporting)

    @staticmethod
    def vectorize_data(data, word_idx, sentence_size, memory_size):
        """
        Vectorize stories and queries.

        If a sentence length < sentence_size, the sentence will be padded with 0's.

        If a story length < memory_size, the story will be padded with empty memories.
        Empty memories are 1-D arrays of length sentence_size filled with 0's.

        The answer array is returned as a one-hot encoding.
        """
        S = []
        Q = []
        A = []
        for story, query, answer in data:
            ss = []
            for i, sentence in enumerate(story, 1):
                ls = max(0, sentence_size - len(sentence))
                ss.append([word_idx[w] for w in sentence] + [0] * ls)

            # take only the most recent sentences that fit in memory
            ss = ss[::-1][:memory_size][::-1]

            # Make the last word of each sentence the time 'word' which 
            # corresponds to vector of lookup table
            for i in range(len(ss)):
                ss[i][-1] = len(word_idx) - memory_size - i + len(ss)

            # pad to memory_size
            lm = max(0, memory_size - len(ss))
            for _ in range(lm):
                ss.append([0] * sentence_size)

            lq = max(0, sentence_size - len(query))
            q = [word_idx[w] for w in query] + [0] * lq

            y = np.zeros(len(word_idx) + 1) # 0 is reserved for nil word
            for a in answer:
                y[word_idx[a]] = 1

            S.append(ss)
            Q.append(q)
            A.append(y)
        return np.array(S), np.array(Q), np.array(A)
