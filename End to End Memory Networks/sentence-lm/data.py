import os, random, socket, collections, pickle, shutil, string, sys, nltk
from pymongo import MongoClient
import numpy as np

class Data:
    def __init__(self, data_path, metadata_path, memory_size = 50, batch_size = 128):
        """ Constructor.

        @param data_path (str) Path to directory where raw data is stored.
        @param metadata_path: (str) Path where metadata (word-to-index mapping, vocabulary size) is stored.
        @param memory_size: (int) Number of context sentences to use.
        @param batch_size: (int)
        """
        self.data_path = data_path
        self.memory_size = memory_size
        self.batch_size = batch_size

        if os.path.exists(metadata_path):
            p = os.path.join(metadata_path, "max_sentence_size.txt")
            with open(p, "r") as f:
                self.max_sentence_size = int(f.read())

            p = os.path.join(metadata_path, "word_to_index.p")
            with open(p, "rb") as f:
                self.word_to_index = pickle.load(f)
        else:
            self._save_metadata(metadata_path)

    def _save_metadata(self, metadata_path):
        os.makedirs(metadata_path)

        words, self.max_sentence_size = _build_metadata(self.data_path)
        self.word_to_index = {word: index + 1 for index, word in enumerate(words)} # Add one since 0 is reserved for padding

        p = os.path.join(metadata_path, "max_sentence_size.txt")
        with open(p, "w") as f:
            f.write(str(self.max_sentence_size))

        p = os.path.join(metadata_path, "word_to_index.p")
        with open(p, "wb") as f:
            pickle.dump(self.word_to_index, f)

    def get_batches(self, data_path):
        """ Generator that yields batches one at a time, to accomodate datasets that won't fit in memory. Data
        is expected to be a collection of documents, each of which is raw text with a single sentence per line.

        @param data_path (str): Path to data directory.

        @return list(tuple(np.ndarray, np.ndarray, np.ndarray))
        """
        if self.word_to_index is None:
            raise ValueError("Must build word-to-index mapping first")

        documents = os.listdir(data_path)
        random.shuffle(documents)

        batch = []
        for document in documents:
            with open(os.path.join(data_path, document), "r") as f:
                sentences = f.readlines()

            for i in range(len(sentences) - (self.memory_size + 2)):
                sentence_context = [self._to_index(sentence) for sentence in sentences[i : i + self.memory_size]]
                query = self._to_index(sentences[i + self.memory_size])
                expected_sentence = self._to_index(sentences[i + self.memory_size + 1])

                batch.append((sentence_context, query, expected_sentence))
                if len(batch) >= self.batch_size:
                    random.shuffle(batch)
                    yield batch
                    batch = []

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

    def _to_index(self, sentence):
        indices = [self.word_to_index[word.lower()] for word in nltk.tokenize.word_tokenize(sentence)]
        indices += [0] * (self.max_sentence_size - len(indices))
        return indices

class MongoConn:
    def __init__(self, ip_addr, port = 27017):
        """ Initializes a connection to a Mongo server located at the specified IP address on the given port.
        
        @param NoneNoneip_addr: (str)
        @param port: (int)
        """
        if not MongoConn._is_ip_address(ip_addr):
            raise ValueError("Not a valid IP address")
        
        try:
            self.client = MongoClient(ip_addr, port)
        except:
            raise ValueError("IP address / port doesn't point to a valid Mongo server")

    def convert(self, data_dir, data_split = (0.85, 0.05, 0.1), collections = None, verbose = True):
        """ Converts the specified collections in the specified databases from the Eigen
        Mongo schema to the format expected by the MNN class.

        @param data_dir: (str) Path to directory where data will be stored.
        @param data_split: (tuple(float)) Unit 3-tuple whose values represent what proportions to split the data into training, validation, and testing, respectively
        @param collections: (dict(str: list(str))) Dictionary with database names as keys and lists of collections in the database as values
        @param verbose: (bool) Whether to display progress information.
        """
        if len(data_split) != 3:
            raise ValueError("Data split proportions must have three values")
        elif sum(data_split) != 1:
            raise ValueError("Data split proportions must be normalized")

        if not os.path.exists(data_dir) or not os.path.isdir(data_dir):
            os.makedirs(data_dir)

        if collections is None:
            collections = {db: self.client[db].collection_names() for db in self.client.database_names()}

        if verbose: print("Reading all documents...")

        # Temporary directory to store data in
        temp_dir = os.path.join("/", "tmp", _random_word(32))
        while os.path.exists(temp_dir):
            temp_dir = os.path.join("/", "tmp", _random_word(32))
        os.makedirs(temp_dir)

        # Extract all data from Mongo database
        num_docs = sum([self.client[db][collection].count() for db in collections for collection in collections[db]])
        paths = []
        for db in collections:
            for collection in collections[db]:
                for document in self.client[db][collection].find():
                    if verbose and len(paths) % 500 == 0:
                        sys.stdout.write("\033[K") # Clear to the end of line
                        print("\tRead %i of %i (%s%%)" % (len(paths), num_docs, str(len(paths) / num_docs * 100)), end = "\r")

                    p = os.path.join(temp_dir, "%i.p" % len(paths)) 
                    paths.append(p)
                    self._convert_document(p, document)
        if verbose:
            sys.stdout.write("\033[K") # Clear to the end of line
            print("\tRead %i of %i (100%%)" % (num_docs, num_docs))

        if verbose: print("Splitting into training, validation, and testing sets")

        # Split data into training, validation, and test data
        data_split = {"train": data_split[0], "validation": data_split[1], "test": data_split[2]}
        for name in ("train", "validation", "test"):
            p = os.path.join(data_dir, name)
            if not os.path.exists(p) or not os.path.isdir(p):
                os.makedirs(p)

            if verbose: print("\tProcessing %s data..." % name)

            end = int(data_split[name] * num_docs)
            for i in range(end):
                if verbose and i % 500 == 0:
                    sys.stdout.write("\033[K") # Clear to the end of line
                    print("\t\tProcessed %i of %i (%s%%)" % (i, end, str(i / end * 100)), end = "\r")

                j = int(random.random() * len(paths))

                index = paths[j].split("/")[-1]
                new_path = os.path.join(p, index)

                shutil.move(paths[j], new_path)
                del paths[j]
            if verbose:
                sys.stdout.write("\033[K") # Clear to the end of line
                print("\tProcessed %i of %i (100%%)" % (end, end))

        # Take care of any remaining files due to round-off error
        p = os.path.join(data_dir, "train")
        for path in paths:
            index = path.split("/")[-1]
            new_path = os.path.join(p, index)
            shutil.move(path, new_path)

        shutil.rmtree(temp_dir)
        if verbose: print("Done.")

    def _convert_document(self, path, document):
        if os.path.exists(path):
            os.remove(path)

        with open(path, "w") as f:
            for paragraph in document["paras"]:
                for sentence_dict in paragraph:
                    sentence = sentence_dict["text"]
                    f.write("%s\n" % sentence)

    @staticmethod
    def _is_ip_address(s):
        try:
            socket.inet_aton(s)
            return True
        except socket.error:
            return False

    listdir_recursive = lambda directory: [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(directory)) for f in fn]

def _build_metadata(path):
    words, max_sentence_size = set([]), -1
    if os.path.isdir(path):
        for sub_file in os.listdir(path):
            sub_path = os.path.join(path, sub_file)
            new_words, new_max_sentence_size = _build_metadata(sub_path)

            words.update(new_words)
            max_sentence_size = max(max_sentence_size, new_max_sentence_size)
    else:
        with open(path, "r") as f:
            for sentence in f.readlines():
                tokens = [word.lower() for word in nltk.tokenize.word_tokenize(sentence)]
                max_sentence_size = max(max_sentence_size, len(tokens))
                words.update(tokens)

    return words, max_sentence_size

def _random_word(length):
    return "".join(random.choice(string.ascii_uppercase) for i in range(length))

if __name__ == "__main__":
    m = MongoConn("10.0.1.180")
    m.convert("/home/ubuntu/data/wikipedia", collections = {"corpora": ["wiki"]})

