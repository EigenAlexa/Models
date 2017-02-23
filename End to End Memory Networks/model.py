import os
import tensorflow as tf
from MemN2N.model import MemN2N
from MemN2N.data import read_data

class MNN:
    DEFAULTS = {
        "edim": 150,
        "lindim": 75,
        "nhop": 6,
        "mem_size": 100,
        "batch_size": 128,
        "nepoch": 100,
        "init_lr": 0.01,
        "init_hid": 0.1,
        "init_std": 0.05,
        "max_grad_norm": 50,
        "data_dir": "data",
        "checkpoint_dir": "checkpoints",
        "data_name" : "ptb",
        "show": False
    }

    def __init__(self, metadata = {}):
        """ Initializes an end-to-end memory network with the architecture and hyperparameters specified in metadata.
        
        @param metadata: dict expecting the following keys:
            edim: (int) internal state dimension
            lindim: (int) linear part of the state
            nhop: (int) number of hops, or layers, in the network
            mem_size: (int) Length of memory matrix
            batch_size: (int) Size of batches to train on
            nepoch: (int) Number of epochs to train for
            init_lr: (float) Initial learning rate
            init_hid: (float) Initial internal state value
            init_std: (float) Standard deviation used to initialize weights (to the Gaussian distribution)
            max_grad_norm: (int) Maximum value the (norm of) gradients can take
            data_dir: (str) Path to the directory containing data to train on
            checkpoint_dir: (str) Path where model is periodically saved at checkpoints during training
            data_name: (str) Name of the data set - the expected format of the data is for training, validation, 
                             and test data to be named <data_name>.train.txt, etc. respectively
            show: (bool) Whether to periodically print progress during training
        """
        self.metadata = {key: (metadata[key] if key in metadata else MNN.DEFAULTS[key]) for key in MNN.DEFAULTS}
        self.mem_net = None
        self.sess = tf.Session()

    def train(self, data_dir, data_name):
        """ Trains the model, using data that was initialized in the constructor based on the given data directory.
        """
        count, word2idx = [], {}
        self.train_data = read_data(os.path.join(data_dir, data_name + ".train.txt"), count, word2idx)
        self.valid_data = read_data(os.path.join(data_dir, data_name + ".valid.txt"), count, word2idx)
        self.test_data = read_data(os.path.join(data_dir, data_name + ".test.txt"), count, word2idx)

        self.metadata["data_dir"] = data_dir
        self.metadata["data_name"] = data_name
        self.metadata["nwords"] = len(word2idx)
        self.metadata["is_test"] = False

        if not os.path.exists(self.metadata["checkpoint_dir"]):
            os.makedirs(self.metadata["checkpoint_dir"])
        
        with self.sess.as_default():
            self.mem_net = MemN2N(self.metadata, self.sess)
            self.mem_net.build_model()
            self.mem_net.run(self.train_data, self.valid_data)

    def test(self):
        """ Tests the model on the test data specified during training.
        """
        if self.mem_net is None:
            raise Exception("Model hasn't been trained yet")

        with self.sess.as_default():
            self.mem_net.is_test = True
            state = self.mem_net.run(self.valid_data, self.test_data)

        return state

    def feed(self, batch):
        if self.mem_net is None:
            raise Exception("Model hasn't been trained yet")

        with self.sess.as_default():
            self.mem_net.is_test = True
            output = self.mem_net.sess.run(self.mem_net.z)

        return output

    def save(self, path = "./MemN2N.model"):
        if self.mem_net is None:
            raise Exception("Model hasn't been trained yet")

        with self.sess.as_default():
            self.mem_net.saver.save(self.mem_net.sess, path, global_step = self.mem_net.step.astype(int))

    def load(self, directory = "./"):
        ckpt = tf.train.get_checkpoint_state(directory)
        if ckpt and ckpt.model_checkpoint_path:
            # self.mem_net.saver.restore(self.mem_net.sess, ckpt.model_checkpoint_path)
            tf.train().Saver().restore(self.mem_net.sess, ckpt.model_checkpoint_path)
        else:
            raise Exception(" [!] Trest mode but no checkpoint found")

    def close():
        self.sess.close()
