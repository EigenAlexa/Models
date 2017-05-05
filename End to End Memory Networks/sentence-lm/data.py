import os, random, socket, collections, pickle, shutil, string, sys, nltk, collections, re
from pymongo import MongoClient
import numpy as np

class Data:
    # TODO implement byte-pair encoding

    def __init__(self, data_path, metadata_path, unk_threshold = 50, memory_size = 50, max_sentence_size = 70, batch_size = 128):
        """ Constructor.

        @param data_path (str) Path to directory where raw data is stored.
        @param metadata_path: (str) Path where metadata (word-to-index mapping, vocabulary size) is stored.
        @param memory_size: (int) Number of context sentences to use.
        @param batch_size: (int)
        """
        self.data_path = data_path
        self.memory_size = memory_size
        self.max_sentence_size = max_sentence_size
        self.batch_size = batch_size
        self.unk_threshold = unk_threshold

        # These will be initialized based on the metadata path
        self.named_entities_index = None
        self.word_frequencies = None

        if os.path.exists(metadata_path):
            self._load_metadata(metadata_path)
        else:
            self._save_metadata(metadata_path)

        words = [word for word in self.word_frequencies.keys() if self.word_frequencies[word] >= self.unk_threshold] + ["<unk>"]
        self.word_to_index = {word: index + 1 for index, word in enumerate(words)} # Add one since 0 is reserved for padding
        self.word_to_index["<unk>"] = len(self.word_to_index) + 1
        self.index_to_word = {self.word_to_index[word]: word for word in self.word_to_index}
        self.named_entities_index_inverse = {self.named_entities_index[named_entity]: named_entity for named_entity in self.named_entities_index}

    def _load_metadata(self, metadata_path):
        p = os.path.join(metadata_path, "max_sentence_size.txt")
        with open(p, "r") as f:
            self.max_sentence_size = int(f.read())

        p = os.path.join(metadata_path, "word_frequencies.p")
        with open(p, "rb") as f:
            self.word_frequencies= pickle.load(f)

        p = os.path.join(metadata_path, "named_entities_index.p")
        with open(p, "rb") as f:
            self.named_entities_index = pickle.load(f)

    def _save_metadata(self, metadata_path):
        os.makedirs(metadata_path)

        class Tokenizer(nltk.tokenize.TreebankWordTokenizer):
            # Simple class that modifies the default NLTK tokenizer to avoid viewing "<" and ">" as delimiters
            def __init__(self):
                self.PARENS_BRACKETS[0] = (re.compile("[\\]\\[\\(\\)\\{\\}]"), " \\g<0> ")

        self.word_frequencies, self.named_entities_index = {}, {}
        self.max_sentence_size, self.num_sentences, self.num_documents = self._extract_metadata(self.data_path, self.word_frequencies, self.named_entities_index, tokenizer)
        self.word_frequencies = dict(self.word_frequencies)

        p = os.path.join(metadata_path, "max_sentence_size.txt")
        with open(p, "w") as f:
            f.write(str(self.max_sentence_size))

        p = os.path.join(metadata_path, "word_frequencies.p")
        with open(p, "wb") as f:
            pickle.dump(self.word_frequencies, f)

        p = os.path.join(metadata_path, "named_entities_index.p")
        with open(p, "wb") as f:
            pickle.dump(self.named_entities_index, f)

    def get_batches(self, data_path):
        """ Generator that yields batches one at a time, to accomodate datasets that won't fit in memory. Data
        is expected to be a collection of documents, each of which is raw text with a single sentence per line.

        @param data_path (str): Path to data directory.

        @return list(tuple(np.ndarray, np.ndarray, np.ndarray))
        """
        documents = os.listdir(data_path)
        random.shuffle(documents)

        batch = []
        for document in documents:
            # print(document)
            with open(os.path.join(data_path, document), "r") as f:
                sentences = f.readlines()
                sentences = [self._to_index(sentence) for sentence in sentences]
                sentences = [sentence for sentence in sentences if len(sentence) <= self.max_sentence_size]

            if len(sentences) < self.memory_size + 2:
                continue

            step, end = min(len(sentences), self.memory_size + 2), max(1, len(sentences) - (self.memory_size + 2))
            if step >= 3:
                for i in range(end):
                    sentence_context = sentences[i : i + step - 2]
                    sentence_context = [[0 for _ in range(self.max_sentence_size)] for _ in range(self.memory_size - (step - 2))] + sentence_context
                    query = sentences[i + step - 2]
                    expected_sentence = sentences[i + step - 1]

                    batch.append((sentence_context, query, expected_sentence))
                    if len(batch) >= self.batch_size:
                        random.shuffle(batch)
                        yield batch
                        batch = []

    def process_raw(self, sentence):
        # Replace named entities
        named_entities = continuous_named_entities(sentence)
        for ne in named_entities:
            lowered_ne = ne.lower()
            if lowered_ne not in self.named_entities_index: 
                sentence = sentence.replace(ne, "<unk>")
            else:
                sentence = sentence.replace(ne, "<NE%i>" % self.named_entities_index[lowered_ne])

        # Tokenize
        indices = self._to_index(sentence)

        # TODO Process with BPE

        return indices

    def unprocess(self, one_hot_sentence):
        # TODO add functionality for undoing BPE

        # Convert to words
        try:
            one_hot_sentence = one_hot_sentence[: list(one_hot_sentence).index(0)]
        except ValueError:
            pass # Sentence is maximum size
        words = [self.index_to_word_lookup(index) for index in one_hot_sentence]

        # Replace named entities
        for i in range(len(words)):
            if words[i].startswith("<ne"): # Found named entity
                index = int(words[i][len("<ne") : -1])
                words[i] = self.index_to_named_entity_lookup(index)

        sentence = " ".join(words)
        return sentence


    def _to_index(self, sentence):
        indices = [self.word_to_index_lookup(word.lower()) for word in nltk.tokenize.word_tokenize(sentence)]
        indices += [0] * (self.max_sentence_size - len(indices))
        return indices

    def word_to_index_lookup(self, word):
        if word not in self.word_to_index:
            word = "<unk>"

        return self.word_to_index[word]

    def index_to_word_lookup(self, index):
        return self.index_to_word[index]

    def named_entity_to_index_lookup(self, named_entity):
        return self.named_entities_index[named_entity]

    def index_to_named_entity_lookup(self, index):
        return self.named_entities_index_inverse[index]

    def _extract_metadata(self, path, freqs, named_entities_index, tokenizer):
        """ Recursively reads all text documents in the given path, extracting word frequencies, the maximum
        sentence size, and the number of batches.
        """
        max_sentence_size = - float("inf")
        if os.path.isdir(path): # Recurse on sub-directories
            for sub_file in os.listdir(path):
                sub_path = os.path.join(path, sub_file)
                sub_max_sentence_size = extract_metadata(sub_path, freqs, named_entities_index, tokenizer)

                # Update
                max_sentence_size = max(max_sentence_size, sub_max_sentence_size)
        else: # Base case
            with open(path, "r") as f:
                try:
                    sentences = f.readlines()
                except UnicodeDecodeError:
                    print("Error: File \"%s\" not encoded with UTF-8" % path)
                    return freqs, max_sentence_size, num_sentences

            for sentence in sentences:
                # Replace named entities with corresponding symbols
                named_entities = continuous_named_entities(sentence)
                for ne in named_entities:
                    lowered_ne = ne.lower()
                    if lowered_ne not in named_entities_index: 
                        # New named entity found, so create a new symbol
                        named_entities_index[lowered_ne] = len(named_entities_index)

                    sentence = sentence.replace(ne, "<NE%i>" % named_entities_index[lowered_ne])

                # Update 
                words = [word.lower() for word in tokenizer.tokenize(sentence)]
                if len(words) <= self.max_sentence_size:
                    freqs += collections.Counter(words)
                max_sentence_size = max(max_sentence_size, len(words))
        
        return max_sentence_size

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

# Utility functions below

def _random_word(length):
    return "".join(random.choice(string.ascii_uppercase) for i in range(length))

def continuous_named_entities(sentence):
    chunks = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sentence)))
    continuous_chunk, curr_chunk = [], []
    for node in chunks:
        if type(node) == nltk.tree.Tree:
            curr_chunk.append(" ".join([token for token, pos in node.leaves()]))
        elif len(curr_chunk) > 0:
            named_entity = " ".join(curr_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
                curr_chunk = []

    return continuous_chunk

# m = MongoConn("10.0.1.180")
# m.convert("/home/ubuntu/data/wikipedia/raw", collections = {"corpora": ["wiki"]})

# import time
# start = time.time()
# data = Data(data_path = "/home/ubuntu/data/wikipedia/",
            # metadata_path = "/home/ubuntu/data/wikipedia/metadata",
            # unk_threshold = 25,
            # memory_size = 50,
            # batch_size = 128)
# end = time.time()
# print("Vocab size:", len(data.word_to_index))
# print("Time: %s seconds" % str(end - start))
