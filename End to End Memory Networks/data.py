import os, random

class Data:
    def __init__(path):
        """
        @param path: (str) Path to directory where data (in the Mongo schema format) is located
        """
        if not os.path.exists(data_path):
            raise ValueError("Data path not found")
        elif not os.path.isdir(data_path):
            raise ValueError("Data path must be a directory")

        self.path = path

    def convert(new_data_dir, new_data_name, data_split = (0.85, 0.05, 0.1)):
        """ Converts data stored in Eigen's conversational format to the form expected by the model. Eigen's 
        format for raw text is as follows:
        
        Each piece of text is a pickled Python dictionary of the form
        {
            "name": name or ID of document,
            "paras": list of paragraphs, each of which is represented as a list containing the paragraph's sentences, each of which stores both the raw text and a tokenized version, ie
                [[(raw sentence, tokenized sentence) for sentence in paragraph] for paragraph in full text]
        }

        The data path is expected to be a directory containing the pickled Python files (with extension .p).

        @param new_data_dir: (str) Path to directory where converted data will be stored
        @param new_data_name: (str) Name of the new data files
        @param data_split: (tuple(float)) Unit 3-tuple whose values represent what proportions to split the data 
                                          into training, validation, and testing, respectively
        """
        if len(data_split) != 3:
            raise ValueError("Data split proportions must have three values")
        elif sum(data_split) != 1:
            raise ValueError("Data split proportions must be normalized")

        if not os.path.exists(new_data_path) or not os.path.isdir(new_data_path):
            os.makedirs(new_data_path)

        data = os.listdir(self.path)
        split_indices = [data_percentage * len(data_files) for data_percentage in data_split]
        random.shuffle(data_files)
        training_data, validation_data, test_data = data[: split_indices[0]], data[split_indices[0] : split_indices[1]], data[split_indices[1] :]

        # Write training data
        with open(os.path.join(new_data_path, new_data_name + ".train.txt"), "w") as f:
            for pickled_file in training_data:
                if pickled_file.endswith(".p"):
                    with open(os.path.join(data_path, pickled_file), "rb") as g:
                        text = _pickle.load(g)
                    
                    for paragraph in text["paras"]:
                        for tokens, sentence in paragraph:
                            f.write("%s\n" % sentence)

        # Write validation data
        with open(os.path.join(new_data_path, new_data_name + ".valid.txt"), "w") as f:
            for pickled_file in validation_data:
                if pickled_file.endswith(".p"):
                    with open(os.path.join(data_path, pickled_file), "rb") as g:
                        text = _pickle.load(g)
                    
                    for paragraph in text["paras"]:
                        for tokens, sentence in paragraph:
                            f.write("%s\n" % sentence)
        
        # Write testing data
        with open(os.path.join(new_data_path, new_data_name + ".test.txt"), "w") as f:
            for pickled_file in validation_data:
                if pickled_file.endswith(".p"):
                    with open(os.path.join(data_path, pickled_file), "rb") as g:
                        text = _pickle.load(g)
                    
                    for paragraph in text["paras"]:
                        for tokens, sentence in paragraph:
                            f.write("%s\n" % sentence)

