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

    def __init__(self, vocab_size, metadata = {}):
        """ Initializes an end-to-end memory network with the architecture and hyperparameters specified in metadata.
        
        @param metadata: (dict) expecting the following keys:
            edim: (int) Dimension of the embedding space
            lindim: (int) The layer outputs are embedded in edim-dimensional space, but are split into an initial linear part followed 
                          by a non-linear part, with only the latter being passed through an activation function; this parameter is up
                          to what index the linear part extends
            nhop: (int) Number of hops, or layers, in the network
            mem_size: (int) Length of memory matrix
            batch_size: (int) Size of batches to train on
            nepoch: (int) Number of epochs to train for
            init_lr: (float) Initial learning rate
            init_hid: (float) Initial internal state value
            init_std: (float) Standard deviation used to initialize weights (to the Gaussian distribution)
            max_grad_norm: (int) Maximum value the (norm of) gradients can take
            checkpoint_dir: (str) Path where model is periodically saved at checkpoints during training
                             and test data to be named <data_name>.train.txt, etc. respectively
            show: (bool) Whether to periodically print progress during training
        """
        self.metadata = {key: (metadata[key] if key in metadata else MNN.DEFAULTS[key]) for key in MNN.DEFAULTS}
        self.metadata["nwords"] = vocab_size
        self.metadata["is_test"] = False

        self.sess = tf.Session(config = tf.ConfigProto(allow_soft_placement = True))
        # self.sess = tf.Session()
        with tf.device("/gpu:0"):
        # if True:
            with self.sess.as_default():
                self.mem_net = MemN2N(self.metadata, self.sess)
                self.mem_net.build_model()

    def train(self, train_data, valid_data, verbose = True):
        """ Trains the model, using data that was initialized in the constructor based on the given data directory.
        """
        if not os.path.exists(self.metadata["checkpoint_dir"]):
            os.makedirs(self.metadata["checkpoint_dir"])
        
        with self.sess.as_default():
            self.mem_net.is_test = False
            self.mem_net.run(train_data, valid_data, verbose = verbose)

    def test(self, valid_data, test_data):
        """ Tests the model on the test data specified during training.
        """
        if self.mem_net is None:
            raise Exception("Model hasn't been trained yet")

        with self.sess.as_default():
            self.mem_net.is_test = True
            state = self.mem_net.run(self.valid_data, self.test_data)

        return state

    def feed(self, data):
        """ Feeds the given data, expected to be a set of MEM_SIZE words, and outputs the models' prediction
        for the next word.

        @param data: (list(int)) List of the last MEM_SIZE words, represented as their one-hot indices.
        
        @return (int) The one-hot index of the predicted word.
        """
        # Prepare inputs
        x = np.ndarray([1, self.metadata["edim"]], dtype = np.float32)
        time = np.ndarray([1, self.metadata["mem_size"]], dtype = np.int32)
        target = np.zeros([1, self.metadata["nwords"]])
        context = np.ndarray(1, self.metadata["mem_size"])

        # Initialize query to constant, time vector to the time steps
        x.fill(self.metadata["init_hid"])
        for t in range(self.metadata["mem_size"]):
            time[:, t].fill(t)

        # Initialize context as the passed in data, reshaped
        for i, word_index in enumerate(data):
            context[0][i] = word_index

        with self.sess.as_default():
            feed = {self.input: x, self.time: time, self.target: target, self.context: context}
            output = self.mem_net.sess.run(self.mem_net.z, feed_dict = feed)

        index = tf.argmax(output, axis = 0)[0]
        return index

    def save(self, path = "./MemN2N.model"):
        if self.mem_net is None:
            raise Exception("Model hasn't been trained yet")

        with self.sess.as_default():
            self.mem_net.saver.save(self.mem_net.sess, path, global_step = self.mem_net.step.astype(int))

    def load(self, directory = "./"):
        ckpt = tf.train.get_checkpoint_state(directory)
        if ckpt and ckpt.model_checkpoint_path:
            tf.train().Saver().restore(self.mem_net.sess, ckpt.model_checkpoint_path)
        else:
            raise Exception(" [!] Trest mode but no checkpoint found")

    def close(self):
        self.sess.close()
