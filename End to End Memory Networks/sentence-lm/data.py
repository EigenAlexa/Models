import os, random
import numpy as np

from sklearn.model_selection import train_test_split

class Data:
    def __init__(self, data_path, metadata_path, memory_size = 50, batch_size = 128):
        """ Constructor.

        @param data_path (str) Path to directory where raw data is stored.
        @param metadata_path: (str) Path where metadata (word-to-index mapping, vocabulary size) is stored.
        @param memory_size: (int) Number of context sentences to use.
        @param batch_size: (int)
        """
        self.data_path = data_path
        self.word_to_index_path = word_to_index_path
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
            self._build_metadata(metadata_path)

    def __build_metadata(self, metadata_path):
        os.makedirs(metadata_path)

        words = set([])
        for document in os.listdir(self.data_path):
            with open(os.path.join(self.data_path, document), "w") as f:
                for sentence in f.readlines():
                    tokens = nltk.tokenize.word_tokenize(sentence)
                    self.max_sentence_size = max(self.max_sentence_size, len(tokens))
                    words.update(tokens)

        self.word_to_index = {word: index for word, index in enumerate(words)}

        p = os.path.join(metadata_path, "max_sentence_size.txt")
        with open(p, "w") as f:
            f.write(self.max_sentence_size)

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

        to_index = lambda sentence: [self.word_to_index(word) for word in nltk.tokenize.word_tokenize(sentence)]
        batch = []
        for document in documents:
            with open(os.path.join(data_path, document), "w") as f:
                sentences = f.read_lines()

            for i in range(len(sentences) - (self.memory_size + 2)):
                sentence_context = [to_index(sentence) for sentence in sentences[i : i + self.memory_size]]
                query = to_index(sentences[i + self.memory_size + 1])
                expected_sentence = to_index(sentences[i + self.memory_size + 2])

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

    def convert(self, data_dir, data_split = (0.85, 0.05, 0.1), collections = None):
        """ Converts the specified collections in the specified databases from the Eigen
        Mongo schema to the format expected by the MNN class.

        @param data_dir: (str) Path to directory where data will be stored.
        @param data_split: (tuple(float)) Unit 3-tuple whose values represent what proportions to split the data into training, validation, and testing, respectively
        @param collections: (dict(str: list(str))) Dictionary with database names as keys and lists of collections in the database as values
        """
        if len(data_storage) != 2:
            raise ValueError()
        elif len(data_split) != 3:
            raise ValueError("Data split proportions must have three values")
        elif sum(data_split) != 1:
            raise ValueError("Data split proportions must be normalized")

        if not os.path.exists(data_dir) or not os.path.isdir(data_dir):
            os.makedirs(data_dir)

        for name in ("train", "valid", "test"):
            p = os.path.join(data_dir, name)
            if not os.path.exists(p) or not os.path.isdir(p)):
                os.makedirs(p)

        if collections is None:
            collections = {db: self.client[db].collection_names() for db in self.client.database_names()}

        documents = [document for db in collections for collection in collections[db] for document in self.client[db][collection].find()]
        random.shuffle(documents)
        split_indices = [int(data_percentage * len(documents)) for data_percentage in data_split]
        for i in range(1, len(data_split)):
            split_indices[i] += split_indices[i - 1]

        training_data, validation_data, test_data = documents[: split_indices[0]], documents[split_indices[0] : split_indices[1]], documents[split_indices[1] :]
        for data, name in zip((training_data, validation_data, test_data), ("train_raw", "valid_raw", "test_raw")):
            for document in data:
                p = os.path.join(data_dir, name, document["name"] + ".txt")
                self._convert_document(p, document)

    def _convert_document(self, path, document):
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

if __name__ == "__main__":
    m = MongoConn("10.0.1.185")
    m.convert("~/data/wikipedia")

