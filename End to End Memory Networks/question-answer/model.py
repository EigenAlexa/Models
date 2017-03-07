import os
import tensorflow as tf
from MemN2N.memn2n import MemN2N

class MNN:
    def __init__(self, vocab_size, max_sent_size, batch_size,
                 mem_size = 20,
                 emb_dim = 150,
                 nhops = 3,
                 nepochs = 100,
                 max_grad_norm = 50,
                 init_lr = 0.01,
                 init_hid = 0.1,
                 init_std = 0.1):
        """ Initializes an end-to-end memory network with the architecture and specified hyperparameters.
            
            @param vocab_size: (int) Size of the dictionary from which the words are drawn.
            @param max_sent_size: (int) Maximum length (in number of words) a sentence can have.
            @param batch_size: (int) Batch size.
            @param mem_size: (int) Size of internal memory, representing how many context sentences are used.
            @param emb_dim: (int) Dimension of the sentence embedding.
            @param nhops: (int) Number of layers or memory hops.
            @param nepochs: (int) Number of epochs to train for.
            @param max_grad_norm: (float) Norm to clip gradients to during gradient descent.
            @param init_lr: (float) Initial learning rate.
            @param init_hid: (float) Value that the dummy query sentence vector is filled with.
            @param init_std: (float) Standard deviation of the Gaussian distribution used to initialize weights.
        """
        self.vocab_size = vocab_size
        self.max_sent_size = max_sent_size
        self.batch_size = batch_size
        self.mem_size = mem_size
        self.emb_dim = emb_dim
        self.nhops = nhops
        self.nepochs = nepochs
        self.max_grad_norm = max_grad_norm
        self.init_lr = init_lr 
        self.init_hid = init_hid
        self.init_std = init_std
        self.sess = tf.Session()

        with self.sess.as_default():
            self.mem_net = MemN2N(batch_size = self.batch_size,
                                  vocab_size = self.vocab_size,
                                  sentence_size = self.max_sent_size,
                                  memory_size = self.mem_size,
                                  embedding_size = self.emb_dim,
                                  hops = self.nhops,
                                  init_lr = self.init_lr,
                                  max_grad_norm = self.max_grad_norm,
                                  nonlin = None,
                                  initializer = tf.random_normal_initializer(stddev = self.init_std),
                                  session = self.sess)

    def train(self, train_data, valid_data, verbose = True):
        """ Trains the model, using data that was initialized in the constructor based on the given data directory.
        """
        with self.sess.as_default():
            self.mem_net.train(train_data, valid_data, self.nepochs, verbose)

    def test(self, test_data):
        """ Tests the model on the test data specified during training, returning the loss and accuracy on the dataset.
        """
        with self.sess.as_default():
            loss = self.mem_net.test(test_data)
            acc = self.mem_net.accuracy(test_data)

        return (loss, acc) 

    def feed(self, sentence_context, query):
        """ Feeds the given data, expected to be a set of MEM_SIZE words, and outputs the models' prediction for the next word.

        @param data: (list(int)) List of the last MEM_SIZE words, represented as their one-hot indices.
        
        @return (int) The one-hot index of the predicted word.
        """
        sentence_context.reshape([1, self.mem_size, self.max_sent_size])
        query.reshape([1, max_sent_size])
        answer = self.mem_net.predict(sentence_context, query)

        return answer

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
