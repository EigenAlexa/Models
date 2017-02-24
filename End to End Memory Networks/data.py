import os, random, socket
from pymongo import MongoClient

class Data:
    def __init__(self, source):
        try:
            self.child = DataDir(source)
            self.convert = self.convert_datadir
        except ValueError:
            try:
                self.child = MongoConn(source)
                self.convert = self.convert_mongo
            except ValueError:
                raise ValueError("Not valid data source")

    def convert_datadir(self, data_source, data_split = (0.85, 0.05, 0.1)):
        self.child.convert(data_source, data_split)

    def convert_mongo(self, data_source, data_split = (0.85, 0.05, 0.1), collections = None):
        self.child.convert(data_source, data_split, collections)

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
            raise ValueError
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

if __name__ == "__main__":
    d = Data("10.0.1.185")
    d.convert(("", ""), collections = {"corpora": ["wiki"]})

