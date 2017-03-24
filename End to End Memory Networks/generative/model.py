import os
import tensorflow as tf
from MemN2N.memn2n import MemN2N
from data import BabiUtils

class MNN:
    def __init__(self,
                 data,
                 batch_size = 32,
                 nepochs = 100,
                 mem_size = 20,
                 emb_dim = 150,
                 nhops = 3,
                 num_rnn_layers = 2,
                 num_lstm_units = 200,
                 lstm_forget_bias = 0.0,
                 max_grad_norm = 50,
                 init_lr = 0.01,
                 init_hid = 0.1,
                 init_std = 0.1):
        """ Initializes an end-to-end memory network with the architecture and specified hyperparameters.
            
            @param data: (data.Data) Data object pointing to the training, validation, and testing data
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
        self.vocab_size = data.vocab_size()
        self.max_sent_size = data.max_sent_size()
        self.word_to_index = data.word2index()

        self.batch_size = batch_size
        self.mem_size = mem_size
        self.emb_dim = emb_dim
        self.nhops = nhops
        self.num_rnn_layers = num_rnn_layers
        self.num_lstm_units = num_lstm_units
        self.lstm_forget_bias = lstm_forget_bias
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
                                  num_rnn_layers = self.num_rnn_layers,
                                  num_lstm_units = self.num_lstm_units,
                                  lstm_forget_bias = self.lstm_forget_bias,
                                  init_lr = self.init_lr,
                                  max_grad_norm = self.max_grad_norm,
                                  nonlin = None,
                                  initializer = tf.random_normal_initializer(stddev = self.init_std),
                                  sess = self.sess)

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
    
    def predict(self, sentences, query):
        """ Feeds raw text data through the model, returning the answer as a word.
        
        @param sentences: (list(str)) Sentence context
        @param query: (str) Query

        @return (str) Answer
        """
        sentences = [BabiUtils._tokenize(sentence) for sentence in sentences]
        # TODO

    def feed(self, sentence_context, query):
        sentence_context = sentence_context.reshape([1, self.mem_size, self.max_sent_size])
        query = query.reshape([1, self.max_sent_size])
        answer = self.mem_net.predict(sentence_context, query)

        return answer

    def save(self, path = "./MemN2N.model"):
        if self.mem_net is None:
            raise Exception("Model hasn't been trained yet")

        with self.sess.as_default():
            self.mem_net.save()

    def load(self, directory = "./"):
        with self.sess.as_default():
            self.mem_net.load()

    def close(self):
        self.sess.close()
