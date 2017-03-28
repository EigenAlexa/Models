# from __future__ import absolute_import
# from __future__ import division

import tensorflow as tf
import numpy as np
# from six.moves import range
# from sklearn import metrics
import random, math, os

def position_encoding(sentence_size, embedding_size):
    """
    Position Encoding for representing sentences described in section 4.1 [1]
    """
    encoding = np.ones((embedding_size, sentence_size), dtype = np.float32)
    ls = sentence_size + 1
    le = embedding_size + 1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i - 1, j - 1] = (i - (embedding_size + 1) / 2) * (j - (sentence_size + 1) / 2)
    encoding = 1 + 4 * encoding / embedding_size / sentence_size 
    encoding[:, -1] = 1.0 # Make position encoding of time words identity to avoid modifying them

    return np.transpose(encoding)

def zero_nil_slot(t, name = None):
    """
    Overwrites the nil_slot (first row) of the input Tensor with zeros.

    The nil_slot is a dummy slot and should not be trained and influence
    the training algorithm.
    """
    with tf.name_scope(name, "zero_nil_slot", [t]) as name:
        t = tf.convert_to_tensor(t, name="t")
        s = tf.shape(t)[1]
        z = tf.zeros(tf.stack([1, s]))
        return tf.concat(axis=0, values=[z, tf.slice(t, [1, 0], [-1, -1])], name=name)

def add_gradient_noise(t, stddev = 1e-3, name = None):
    """
    Adds gradient noise as described in http://arxiv.org/abs/1511.06807 [2].

    The input Tensor `t` should be a gradient.

    The output will be `t` + gaussian noise.

    0.001 was said to be a good fixed value for memory networks [2].
    """
    with tf.name_scope(name, "add_gradient_noise", [t, stddev]) as name:
        t = tf.convert_to_tensor(t, name = "t")
        gn = tf.random_normal(tf.shape(t), stddev = stddev)
        return tf.add(t, gn, name = name)

class MemN2N(object):
    """End-To-End Memory Network."""
    def __init__(self,
                 batch_size,
                 vocab_size,
                 sentence_size,
                 memory_size,
                 embedding_size,
                 hops = 3,
                 num_rnn_layers = 2,
                 num_lstm_units = 200,
                 lstm_forget_bias = 0.0,
                 init_lr = 0.01,
                 max_grad_norm = 40.0,
                 nonlin = None,
                 initializer = tf.random_normal_initializer(stddev = 0.1),
                 encoding = position_encoding,
                 sess = tf.Session(),
                 name = "MemN2N"):
        """ Constructor.

        @param batch_size: (int) Batch size.
        @param vocab_size: (int) Number of distinct words in data, including the nil word, whose one-hot encoding is the zero vector.
        @param sentence_size: (int) Maximum number of words in any sentence in the data.
        @param memory_size: (int) Maximum number of context sentences used.
        @param embedding_size: (int) Dimension of word embedding vectors, and if position encoding is used, sentence representation dimension as well.
        @param hops: (int) Number of memory hops to use.
        @param init_lr: (float) Initial learning rate.
        @param max_grad_norm: (float): Maximum value gradients are clipped to during backpropagation.
        @param nonlin: (func float -> float) Optional non-linear function to apply to output of last memory hop.
        @param initializer: (Tensorflow func) Initializer function for parameters.
        @param encoding: (func) Which kind of sentence representation to use; should return an encoding vector.
        @param sess: (tf.Session) Active tensorflow session to use.
        @param name: (str) Name of model.
        """

        self._batch_size = batch_size
        self._vocab_size = vocab_size
        self._sentence_size = self._num_rnn_steps = sentence_size
        self._memory_size = memory_size
        self._embedding_size = embedding_size
        self._hops = hops
        self._num_rnn_layers = num_rnn_layers
        self._num_lstm_units = num_lstm_units
        self._lstm_forget_bias = lstm_forget_bias
        self._init_lr = init_lr
        self._max_grad_norm = max_grad_norm
        self._nonlin = nonlin
        self._initializer = initializer
        self._sess = sess 
        self._name = name
        self._checkpoint_dir = "./checkpoints"

        # Sentence encoding vector
        self._encoding = tf.constant(encoding(self._sentence_size, self._embedding_size), name = "encoding")

        # Set up the model architecture
        self._build_inputs()
        self._build_params()
        self._build_model()
        self._build_training()
        self._predict_op = tf.argmax(self._output, 1, name = "predict_op") # Model answer is one-hot vector

        self._sess.run(tf.global_variables_initializer())

    def _build_inputs(self):
        # Create nodes for expected inputs
        self._sentence_context = tf.placeholder(tf.int32, [None, self._memory_size, self._sentence_size], name = "sentence_context") # Memory matrix
        self._queries = tf.placeholder(tf.int32, [None, self._sentence_size], name = "queries") # Query sentence
        self._answers = tf.placeholder(tf.int32, [None, self._vocab_size], name = "answers") # Answer
        self._learning_rate = tf.placeholder(tf.float32)

    def _build_params(self):
        # Create nodes for model parameters, which are the embedding matrices
        with tf.variable_scope(self._name):
            nil_word_slot = tf.zeros([1, self._embedding_size])
            A = tf.concat(axis = 0, values = [nil_word_slot, self._initializer([self._vocab_size - 1, self._embedding_size])])
            C = tf.concat(axis = 0, values = [nil_word_slot, self._initializer([self._vocab_size - 1, self._embedding_size])])

            # Adjacent weight sharing - each embedding in self.C is the output embedding for layer l and memory embedding for layer l + 1
            self.A_1 = tf.Variable(A, name = "A") # Initial memory embedding
            self.C = []

            # Build output embedding matrices
            for hopn in range(self._hops):
                with tf.variable_scope('hop_{}'.format(hopn)):
                    self.C.append(tf.Variable(C, name = "C"))

            # Answer prediction weight matrix
            self.W = tf.Variable(self._initializer([self._embedding_size, self._embedding_size]), name = "W")

            # Linear mapping for layer output (not necessary with adjacent weight sharing)
            # self.H = tf.Variable(self._initializer([self._embedding_size, self._embedding_size]), name = "H") 

        self._nil_vars = set([self.A_1.name] + [x.name for x in self.C])

        # Build weights for output of RNN
        self._rnn_W = tf.Variable(self._initializer([self._num_lstm_units, self._vocab_size]))
        self._rnn_b = tf.Variable(self._initializer([self._vocab_size]))

    def _build_model(self):
        # Feedforward input through the model
        with tf.variable_scope(self._name):
            # Use A_1 for the query embedding as per Adjacent Weight Sharing
            q_emb = tf.nn.embedding_lookup(self.A_1, self._queries)
            u = tf.reduce_sum(q_emb * self._encoding, axis = 1)

            for hopn in range(self._hops):
                u = self._build_hop(hopn, u)

                # Nonlinearity
                if self._nonlin:
                    u = nonlin(u)

            # Use last C for output (transposed)
            with tf.variable_scope('hop_{}'.format(self._hops)):
                self._answer = tf.matmul(u, self.W)

            self._build_rnn()

    def _build_hop(self, hopn, prev_u):
        # Store context sentences in memory, encoded with memory embedding
        if hopn == 0:
            m_emb_A = tf.nn.embedding_lookup(self.A_1, self._sentence_context) # Get word embeddings
            m_A = tf.reduce_sum(m_emb_A * self._encoding, axis = 2) # Convert to sentence representation
        else:
            with tf.variable_scope('hop_{}'.format(hopn - 1)):
                m_emb_A = tf.nn.embedding_lookup(self.C[hopn - 1], self._sentence_context) # Adjacent weight sharing - last output embedding becomes memory embedding
                m_A = tf.reduce_sum(m_emb_A * self._encoding, axis = 2) # Convert to sentence representation

        # Compute probability vector, by passing the cosine similarities between (encoded) query and (encoded) memory vectors
        u_temp = tf.transpose(tf.expand_dims(prev_u, -1), [0, 2, 1]) # Hack to get around no reduce_dot
        dotted = tf.reduce_sum(m_A * u_temp, 2)
        p = tf.nn.softmax(dotted)

        # Compute output vectors (context sentences encoded with output embedding)
        with tf.variable_scope('hop_{}'.format(hopn)):
            m_emb_C = tf.nn.embedding_lookup(self.C[hopn], self._sentence_context)
        m_C = tf.reduce_sum(m_emb_C * self._encoding, axis = 2)

        p_temp = tf.transpose(tf.expand_dims(p, -1), [0, 2, 1])
        c_temp = tf.transpose(m_C, [0, 2, 1])
        o = tf.reduce_sum(c_temp * p_temp, 2) # Response vector

        u = prev_u + o # Layer output
        # u = tf.matmul(prev_u, self.H) + o # Layer output with linear mapping (not necessary with adjacent weight sharing)
        
        return u

    def _build_rnn(self):
        # Build LSTM cells
        lstm = lambda: tf.contrib.rnn.BasicLSTMCell(self._num_lstm_units, self._lstm_forget_bias, state_is_tuple = True)
        cell = tf.contrib.rnn.MultiRNNCell([lstm() for _ in range(self._num_rnn_layers)])

        # Propagate answer through unrolled LSTM cells
        rnn_inputs = [self._answer for _ in range(self._num_rnn_steps)]
        layer_outputs, _ = tf.contrib.rnn.static_rnn(cell, rnn_inputs, dtype = tf.float32)

        layer_outputs = tf.transpose(layer_outputs, [1, 0, 2])
        self._output = tf.einsum("ijk,kl->ijl", layer_outputs, self._rnn_W) + self._rnn_b

    def _build_training(self):
        # Cross entropy loss
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = first_words, labels = tf.cast(self._answers, tf.float32), name = "cross_entropy")
        cross_entropy_sum = tf.reduce_sum(cross_entropy, name = "cross_entropy_sum")
        self._loss_op = cross_entropy_sum

        # Gradient pipeline
        self._opt = tf.train.GradientDescentOptimizer(learning_rate = self._learning_rate)
        grads_and_vars = self._opt.compute_gradients(self._loss_op)

        # Clip gradients
        grads_and_vars = [(tf.clip_by_norm(g, self._max_grad_norm), v) for g, v in grads_and_vars]
        grads_and_vars = [(add_gradient_noise(g), v) for g, v in grads_and_vars]

        # Add degenerate gradients for null-paddings
        nil_grads_and_vars = []
        for g, v in grads_and_vars:
            if v.name in self._nil_vars:
                nil_grads_and_vars.append((zero_nil_slot(g), v))
            else:
                nil_grads_and_vars.append((g, v))

        self._train_op = self._opt.apply_gradients(nil_grads_and_vars, name = "train_op")

    def train(self, train_data, valid_data, nepochs = -1, verbose = True):
        """ Trains the model on the given training and validation data.

        @param train_data: (list(tuple(np.ndarray, np.ndarray, np.ndarray))) where the three numpy arrays represent a story, query, and 
            answer. Each story has shape (memory_size, max_sentence_length) and is a list of sentences, each query has shape (max_sentence_length,),
            and each answer has shape (vocab_size,). Each sentence is an array of words, where each word is represented by its one-hot index. The
            answer is also a word, but is expanded to its full one-hot vector form rather than just the index.
        @param valid_data: (list(tuple(np.ndarray, np.ndarray, np.ndarray))) Same as training data, but used
            for validation.
        @param nepochs: (int) How many epochs to train for, or -1 to repeat until convergence (defined as 5 epochs without improvement).
        """
        learning_rate = self._init_lr
        old_valid_loss = float("inf")

        count, epoch = 0, 0
        while True:
            if nepochs != -1 and epoch > nepochs:
                break

            train_loss = self._train_batches_SGD(train_data, self._batch_size, learning_rate)
            valid_loss = self.test(valid_data)

            # Learning rate annealing
            if valid_loss > old_valid_loss * 0.9999:
                learning_rate *= 2 / 3
                count += 1
            else:
                count = 0

            if verbose and epoch % 10 == 0:
                valid_acc = self.accuracy(valid_data)
                print("Epoch %i" % epoch)
                print("\tTraining loss: %s" % str(train_loss))
                print("\tValidation loss: %s" % str(valid_loss))
                print("\tValidation accuracy: %s" % str(valid_acc))
                print("\tLearning rate: %s" % str(learning_rate))
                print()

            if nepochs == -1 and count > 5:
                break

            old_valid_loss = valid_loss
            epoch += 1

        if verbose:
            valid_acc = self.accuracy(valid_data)
            print("Final validation accuracy: %s" % str(valid_acc))

    def _train_batches_SGD(self, data, batch_size, learning_rate):
        """ Trains the model on the given data.
        """
        num_batches = int(math.ceil(len(data) / batch_size))
        batch_indices = list(range(0, len(data) - batch_size, batch_size))
        random.shuffle(batch_indices)

        loss = 0
        for batch_index in batch_indices:
        # for _ in range(num_batches):
            # # Stochastically build batch
            # batch_index = random.randrange(batch_size, len(data))
            batch = data[batch_index : batch_index + batch_size]

            sentence_context, queries, answers = zip(*batch)
            sentence_context, queries, answers = np.array(sentence_context), np.array(queries), np.array(answers)
            
            loss += self._train_batch(sentence_context, queries, answers, learning_rate)

        return loss / (num_batches * batch_size) # Return average loss

    def _train_batch(self, sentence_context, queries, answers, learning_rate):
        """Runs the training algorithm over the passed batch (list of stories, where a story is a list
        of sentences, where a sentence is a list of words, where a word is encoded with its index).

        Args:
            sentence_context: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)
            answers: Tensor (None, vocab_size)

        Returns:
            loss: floating-point number, the loss computed for the batch
        """
        # Prepare inputs
        feed_dict = {
            self._sentence_context: sentence_context,
            self._queries: queries,
            self._answers: answers,
            self._learning_rate: learning_rate
        }

        loss, _ = self._sess.run([self._loss_op, self._train_op], feed_dict = feed_dict)
        return loss 

    def test(self, data):
        """ Runs the model on the data and returns the loss, without training the model.
        """
        sentence_context, queries, answers = zip(*data)
        sentence_context, queries, answers = np.array(sentence_context), np.array(queries), np.array(answers)

        feed = {self._sentence_context: sentence_context, self._queries: queries, self._answers: answers}
        loss = self._sess.run(self._loss_op, feed_dict = feed)
        return loss / len(data)
    
    def accuracy(self, data):
        sentence_context, queries, answers = zip(*data)
        sentence_context, queries, answers = np.array(sentence_context), np.array(queries), np.array(answers)
        labels = np.argmax(answers, axis = 1)

        outputs = self._sess.run(self._output, feed_dict = {self._sentence_context: sentence_context, self._queries: queries})
        first_words = outputs[:, 0, :]
        predictions = np.argmax(first_words, axis = 1)
        # predictions = self.predict(sentence_context, queries)
        # acc = metrics.accuracy_score(predictions, labels)
        acc = np.sum(predictions == labels) / len(predictions)

        return acc

    def predict(self, sentence_context, queries):
        """Predicts answers as one-hot encoding.

        Args:
            sentence_context: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)

        Returns:
            answers: Tensor (None, vocab_size)
        """
        feed_dict = {self._sentence_context: sentence_context, self._queries: queries}
        return self._sess.run(self._predict_op, feed_dict = feed_dict)
    
    def save(self, index = None):
        if not os.path.exists(self._checkpoint_dir) or not os.path.isdir(self._checkpoint_dir):
            os.makedirs(self._checkpoint_dir)

        if index is None:
            index = len(os.listdir(self._checkpoint_dir))

        model_dir = "model_%i" % index
        os.makedirs(os.path.join(self._checkpoint_dir, model_dir))
        model_file = os.path.join(self._checkpoint_dir, model_dir, self._name)

        saver = tf.train.Saver()
        saver.save(self._sess, model_file)

    def load(self, index = None):
        if index is None:
            index = len(os.listdir(self._checkpoint_dir)) - 1

        model_dir = "model_%i" % index
        model_file = os.path.join(self._checkpoint_dir, model_dir, self._name)

        saver = tf.train.Saver()
        saver.restore(self._sess, model_file)
