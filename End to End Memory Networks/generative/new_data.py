import os, random, socket, collections
from pymongo import MongoClient

class Data:
    def __init__(self, source, raw = True):
        """ Wrapper class for handling data.
        
        @param source: (str) Path to data directory or file.
        @param raw: (bool) Whether the data is raw and needs processing, or is already in the expected format.
        """
        if raw:
            self.read_data = None
            try:
                self.child = DataDir(source)
                self.convert = self._convert_datadir
            except ValueError:
                try:
                    self.child = MongoConn(source)
                    self.convert = self._convert_mongo
                except ValueError:
                    raise ValueError("Not valid data source")
        else:
            self.convert = None
            self.child = ProcessedData(source)
            self.read_data = self._read_data

    def _convert_datadir(self, data_source, data_split = (0.85, 0.05, 0.1)):
        if self.convert != self._convert_datadir:
            raise Exception("Wrong data class")

        self.child.convert(data_source, data_split)

    def _convert_mongo(self, data_source, data_split = (0.85, 0.05, 0.1), collections = None):
        if self.convert != self._convert_mongo:
            raise Exception("Wrong data class")

        self.child.convert(data_source, data_split, collections)

    def _read_data(self):
        if self.read_data != self._read_data:
            raise Exception("Wrong data type")

        return self.child.read_data()

class DataDir(Data):
    def __init__(self, path):
        """
        @param path: (str) Path to directory where data (in the Mongo schema format) is located
        """
        if not os.path.exists(path):
            raise ValueError("Data path not found")
        elif not os.path.isdir(path):
            raise ValueError("Data path must be a directory")

        self.path = path

    def convert(self, data_storage, data_split = (0.85, 0.05, 0.1)):
        """ Converts data stored in Eigen's conversational format to the form expected by the model. Eigen's 
        format for raw text is as follows:
        
        Each piece of text is a pickled Python dictionary of the form
        {
            "name": name or ID of document,
            "paras": list of paragraphs, each of which is represented as a list containing the paragraph's sentences, each of which stores both the raw text and a tokenized version, ie
                [[(raw sentence, tokenized sentence) for sentence in paragraph] for paragraph in full text]
        }

        The data path is expected to be a directory containing the pickled Python files (with extension .p).

        @param data_storage: (tuple(str, str)) Tuple containing path to directory where converted files will be stored and the name of the new data files
        @param data_split: (tuple(float, float, float)) Unit 3-tuple whose values represent what proportions to split the data into training, validation, and testing, respectively
        """
        if len(data_storage) != 2:
            raise ValueError
        elif len(data_split) != 3:
            raise ValueError("Data split proportions must have three values")
        elif sum(data_split) != 1:
            raise ValueError("Data split proportions must be normalized")

        new_data_dir, new_data_name = data_storage

        if not os.path.exists(new_data_path) or not os.path.isdir(new_data_path):
            os.makedirs(new_data_path)

        data = listdir_recursive(self.path)
        split_indices = [int(data_percentage * len(data_files)) for data_percentage in data_split]
        random.shuffle(data_files)
        training_data, validation_data, test_data = data[: split_indices[0]], data[split_indices[0] : split_indices[1]], data[split_indices[1] :]

        for pickled_files, name in zip((training_data, validation_data, test_data), ("train", "valid", "test")):
            self._convert_data(os.path.join(new_data_dir, new_data_name + ".%s.txt" % name), pickled_files)

    def _convert_data(self, path, pickled_files):
        with open(path, "w") as f:
            for pickled_file in pickled_files:
                if pickled_file.endswith(".p"):
                    with open(os.path.join(data_path, pickled_file), "rb") as g:
                        text = _pickle.load(g)
                    
                    for paragraph in text["paras"]:
                        for sentence_dict in paragraph:
                            sentence = sentence_dict["text"]
                            f.write("%s\n" % sentence)

class MongoConn(Data):
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

    def convert(self, data_storage, data_split = (0.85, 0.05, 0.1), collections = None):
        """ Converts the specified collections in the specified databases from the Eigen
        Mongo schema to the format expected by the MNN class.

        @param data_storage: (tuple(str, str)) Tuple containing path to directory where converted files will be stored and the name of the new data files
        @param data_split: (tuple(float)) Unit 3-tuple whose values represent what proportions to split the data into training, validation, and testing, respectively
        @param collections: (dict(str: list(str))) Dictionary with database names as keys and lists of collections in the database as values
        """
        if len(data_storage) != 2:
            raise ValueError()
        elif len(data_split) != 3:
            raise ValueError("Data split proportions must have three values")
        elif sum(data_split) != 1:
            raise ValueError("Data split proportions must be normalized")

        new_data_dir, new_data_name = data_storage

        if not os.path.exists(new_data_dir) or not os.path.isdir(new_data_dir):
            os.makedirs(new_data_dir)

        if collections is None:
            collections = {db: self.client[db].collection_names() for db in self.client.database_names()}

        documents = [document for db in collections for collection in collections[db] for document in self.client[db][collection].find()]
        random.shuffle(documents)
        split_indices = [int(data_percentage * len(documents)) for data_percentage in data_split]
        for i in range(1, len(data_split)):
            split_indices[i] += split_indices[i - 1]

        training_data, validation_data, test_data = documents[: split_indices[0]], documents[split_indices[0] : split_indices[1]], documents[split_indices[1] :]
        for data, name in zip((training_data, validation_data, test_data), ("train", "valid", "test")):
            for document in data:
                self._convert_document(os.path.join(new_data_dir, new_data_name + ".%s.txt" % name), document)

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

class ProcessedData:
    def __init__(self, source):
        if len(source) != 2:
            raise ValueError()

        data_dir, data_name = source
        if not os.path.isdir(data_dir):
            raise ValueError("Data source must be a directory")

        data_sources = {}
        for name in ("train", "valid", "test"):
            path = os.path.join(data_dir, "%s.%s.txt" % (data_name, name))
            if not os.path.exists(path):
                raise ValueError("%s data file not found" % name)
            
            data_sources[name] = path

        self.data_sources = data_sources

    def read_data(self):
        """ Read text stored at the data source into a dataset for internal use.
        
            @return data: (list(int)) List of words directly from the text, with each word represented with its index in its one-hot
                                      representation. Sentences are separated with a special <eos> word.
        """
        count, word2idx = [['<eos>', 0]], {"<eos>": 0}
        data = []
        for i, key in enumerate(("train", "valid", "test")):
            words, word2idx = self._read_data_helper(self.data_sources[key], count, word2idx)
            data.append(words)

        return (tuple(data), word2idx)

    def _read_data_helper(self, source, count, word2idx):
        if os.path.isfile(source):
            with open(source) as f:
                lines = f.readlines()
        else:
            raise ValueError("Data not found")

        # Get words
        words = []
        for line in lines:
            words += line.split()

        # Get word counts
        count[0][1] += len(lines)
        count.extend(collections.Counter(words).most_common())

        # Assign each word a unique index for its one-hot representation
        for word, _ in count:
            if word not in word2idx:
                word2idx[word] = len(word2idx)

        # Build data, the full text with words converted to their one-hot indices
        data = []
        for line in lines:
            for word in line.split():
                index = word2idx[word]
                data.append(index)
            data.append(word2idx['<eos>'])

        return (data, word2idx)

