import os, math, random
import numpy as np
import tensorflow as tf

# Memory-recurrent network
class MNN:
    def __init__(self,
                 data,
                 batch_size = 32,
                 memory_size = 20,
                 embedding_dim = 150,
                 nhops = 3,
                 num_rnn_layers = 2,
                 num_lstm_units = 200,
                 lstm_forget_bias = 0.0,
                 rnn_dropout_keep_prob = 1.0,
                 init_lr = 0.01,
                 max_grad_norm = 40.0,
                 nonlin = None,
                 initializer = tf.random_normal_initializer(stddev = 0.1),
                 sess = tf.Session(),
                 name = "MemN2N"):
        """ Constructor.

        @param data: (data.Data) Data object pointing to the training, validation, and testing data
        @param batch_size: (int) Batch size.
        @param memory_size: (int) Maximum number of context sentences used.
        @param embedding_dim: (int) Dimension of word embedding vectors, and if position encoding is used, sentence representation dimension as well.
        @param nhops: (int) Number of memory nhops to use.
        @param num_rnn_layers: (int) Number of layers in the answer RNN
        @param num_lstm_units: (int) Number of LSTM units per layer in the answer RNN
        @param lstm_forget_bias: (float) Forget bias of each LSTM unit in answer RNN
        @param rnn_dropout_keep_prob: (float) Probability that a node in the answer RNN is kept during training with dropout
        @param init_lr: (float) Initial learning rate.
        @param max_grad_norm: (float): Maximum value gradients are clipped to during backpropagation.
        @param nonlin: (func float -> float) Optional non-linear function to apply to output of last memory hop.
        @param initializer: (Tensorflow func) Initializer function for parameters.
        @param sess: (tf.Session) Active tensorflow session to use.
        @param name: (str) Name of model.
        """
        self._vocab_size = data.vocab_size()
        self._sentence_size = self._num_rnn_steps = data.max_sent_size()
        self._word_to_index = data.word2index()
        self._index_to_word = {index: word for word, index in self._word_to_index.items()}

        # Hyperparameters
        self._batch_size = batch_size
        self._memory_size = memory_size
        self._embedding_size = embedding_dim
        self._hops = nhops
        self._num_rnn_layers = num_rnn_layers
        self._num_lstm_units = num_lstm_units
        self._lstm_forget_bias = lstm_forget_bias
        self._rnn_dropout_keep_prob = rnn_dropout_keep_prob
        self._init_lr = init_lr
        self._max_grad_norm = max_grad_norm
        self._nonlin = nonlin
        self._initializer = initializer

        # TODO delete
        print("vocab_size:", self._vocab_size)
        print("sentence_size:", self._sentence_size)
        print("batch_size:", self._batch_size)
        print("memory_size:", self._memory_size)
        print("embedding_dim:", self._embedding_size)
        print("hops:", self._hops)
        print("num_rnn_layers:", self._num_rnn_layers)
        print("num_lstm_units:", self._num_lstm_units)

        self._sess = sess
        self._name = name
        self._checkpoint_dir = "./checkpoints" 
        self._encoding = tf.constant(position_encoding(self._sentence_size, self._embedding_size), name = "encoding") # Sentence encoding vector

        # Set up the model architecture
        self._build_inputs()
        self._build_params()
        self._build_model()
        self._build_training()
        self._predict_op = self._output

        self._sess.run(tf.global_variables_initializer())

    def _build_inputs(self):
        # Create nodes for expected inputs
        self._sentence_context = tf.placeholder(tf.int32, [None, self._memory_size, self._sentence_size], name = "sentence_context") # Memory matrix
        self._queries = tf.placeholder(tf.int32, [None, self._sentence_size], name = "queries") # Query sentence
        self._expected_sentences = tf.placeholder(tf.int32, [None, self._sentence_size], name = "answers") # Answer
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
        self._rnn_W = tf.Variable(self._initializer([self._num_lstm_units, self._vocab_size]), name = "rnn_W")
        self._rnn_b = tf.Variable(self._initializer([self._vocab_size]), name = "rnn_b")

    def _build_model(self):
        with tf.variable_scope(self._name):
            # Use A_1 for the query embedding as per Adjacent Weight Sharing
            q_emb = tf.nn.embedding_lookup(self.A_1, self._queries)
            u = tf.reduce_sum(q_emb * self._encoding, axis = 1)

            # Build feedforward memory hops
            for hopn in range(self._hops):
                u = self._build_hop(hopn, u)

                # Nonlinearity
                if self._nonlin:
                    u = nonlin(u)

            # Use last C for output (transposed)
            with tf.variable_scope('hop_{}'.format(self._hops)):
                self._answer = tf.matmul(u, self.W)

            # Pass output of memory hops through RNN
            self._build_rnn()

    def _build_hop(self, hopn, prev_u):
        # Store context sentences in memory, encoded with memory embedding
        if hopn == 0:
            m_emb_A = tf.nn.embedding_lookup(self.A_1, self._sentence_context) # Get word embeddings
            m_A = tf.reduce_sum(m_emb_A * self._encoding, axis = 2) # Convert to sentence representation
        else:
            with tf.variable_scope('hop_{}'.format(hopn - 1)):
                m_emb_A = tf.nn.embedding_lookup(self.C[hopn - 1], self._sentence_context) # Adjacent weight sharing - previous output embedding becomes memory embedding
                m_A = tf.reduce_sum(m_emb_A * self._encoding, axis = 2) # Convert to sentence representation

        # Compute probability vector, by passing the cosine similarities between (encoded) query and (encoded) memory vectors
        u_temp = tf.transpose(tf.expand_dims(prev_u, -1), [0, 2, 1]) # Hack to get around no reduce_dot
        dotted = tf.reduce_sum(m_A * u_temp, 2)
        p = tf.nn.softmax(dotted)

        # Compute output vectors (context sentences encoded with output embedding)
        with tf.variable_scope('hop_{}'.format(hopn)):
            m_emb_C = tf.nn.embedding_lookup(self.C[hopn], self._sentence_context)
        m_C = tf.reduce_sum(m_emb_C * self._encoding, axis = 2)

        # Compute response vector o
        p_temp = tf.transpose(tf.expand_dims(p, -1), [0, 2, 1])
        c_temp = tf.transpose(m_C, [0, 2, 1])
        o = tf.reduce_sum(c_temp * p_temp, 2)

        # Layer output
        u = prev_u + o 
        return u

    def _build_rnn(self):
        # Build LSTM cells
        lstm = lambda: tf.contrib.rnn.BasicLSTMCell(self._num_lstm_units, self._lstm_forget_bias, state_is_tuple = True)
        if self._rnn_dropout_keep_prob < 1.0:
            normal_lstm = lstm
            lstm = lambda: tf.contrib.rnn.DropoutWrapper(normal_lstm())

        cell = tf.contrib.rnn.MultiRNNCell([lstm() for _ in range(self._num_rnn_layers)])

        # Propagate answer through unrolled LSTM cells
        rnn_inputs = [self._answer for _ in range(self._num_rnn_steps)]
        if self._rnn_dropout_keep_prob < 1.0:
            rnn_inputs = [tf.nn.dropout(rnn_input, keep_prob = self._rnn_dropout_keep_prob) for rnn_input in rnn_inputs]
        layer_outputs, _ = tf.contrib.rnn.static_rnn(cell, rnn_inputs, dtype = tf.float32)
        if self._rnn_dropout_keep_prob < 1.0:
            layer_outputs = [tf.nn.dropout(output, self._rnn_dropout_keep_prob) for output in layer_outputs]

        # Pass RNN outputs through final linear transform into word manifold
        layer_outputs = tf.transpose(layer_outputs, [1, 0, 2])
        self._output = tf.einsum("ijk,kl->ijl", layer_outputs, self._rnn_W) + self._rnn_b # shape (batch_size, sentence_size, vocab_size)

    def _build_training(self):
        # Loss
        self._loss_op = self._loss_function()

        # Gradient pipeline
        self._opt = tf.train.GradientDescentOptimizer(learning_rate = self._learning_rate)
        grads_and_vars = self._opt.compute_gradients(self._loss_op)

        # Clip gradients
        grads_and_vars = [(tf.clip_by_norm(g, self._max_grad_norm), v) for g, v in grads_and_vars]
        grads_and_vars = [(add_gradient_noise(g), v) for g, v in grads_and_vars] # Sprinkle in some Gaussian noise

        # Add degenerate gradients nil-paddings
        nil_grads_and_vars = []
        for g, v in grads_and_vars:
            if v.name in self._nil_vars:
                nil_grads_and_vars.append((zero_nil_slot(g), v))
            else:
                nil_grads_and_vars.append((g, v))

        self._train_op = self._opt.apply_gradients(nil_grads_and_vars, name = "train_op")

    def _loss_function(self):
        logits = tf.cast(tf.argmax(self._output, axis = 2), tf.float32)
        # expected_sentences = tf.one_hot(self._expected_sentences, depth = self._vocab_size)

        # Approach 1 (default TF sequence-to-sequence loss)
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [self._expected_sentences], [tf.ones([self._batch_size])])

        # Approach 2 (cross entropy between raw sentence vectors)
        # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, self._expected_sentences)
        # loss = tf.reduce_sum(cross_entropy)

        # Approach 3 (cross entropy between embedded sentence representations)
        # TODO

        return loss

    def train(self, train_data, valid_data, nepochs = -1, verbose = True):
        """ Trains the model on the given training and validation data.

        @param train_data (list(tuple(story, query, expected_sentence)) where story is list with shape (memory_size, sentence_size), query
            is list of length sentence_size, and expected_sentence is list of length sentence_size, each containing integers): Each sentence
            is an array of words, where each word is represented with its one-hot index. Alternatively, train_data might be a generator
            that yields batches (sub-lists of the 3-tuples) one at a time, in case the full data set doesn't fit in memory.
        @param valid_data: (list(tuple(story, query, expected_sentence))) Same as training data, but used
            for validation.
        @param nepochs: (int) How many epochs to train for, or -1 to repeat until convergence (defined as 5 epochs without improvement).
        """
        with self._sess.as_default():
            learning_rate = self._init_lr
            old_valid_loss = float("inf")

            # Train until convergence or nepochs
            count, epoch = 0, 0
            while True:
                if nepochs != -1 and epoch > nepochs:
                    break

                # Train epoch
                train_loss = self._train_batches_SGD(train_data, learning_rate)
                valid_loss = self.test(valid_data)

                # Learning rate annealing
                if valid_loss > old_valid_loss * 0.9999:
                    learning_rate *= 2 / 3
                    count += 1
                else:
                    count = 0

                if verbose and epoch % 10 == 0:
                    print("Epoch %i" % epoch)
                    print("\tTraining loss: %s" % str(train_loss))
                    print("\tValidation loss: %s" % str(valid_loss))
                    print("\tLearning rate: %s" % str(learning_rate))
                    print()

                # If converged, stop training
                if nepochs == -1 and count > 5:
                    break

                old_valid_loss = valid_loss
                epoch += 1

    def _train_batches_SGD(self, data, learning_rate):
        loss, num_batches = 0.0, 0

        # Prepare batches
        if type(data) == list:
            batches = [data[i : i + self._batch_size] for i in range(0, len(data) - self._batch_size, self._batch_size)]
            random.shuffle(batches)
        else: # data is a generator that yields batches, to save memory
            batches = data

        # Train each batch
        for batch in batches:
            sentence_context, queries, expected_sentences = zip(*batch)
            loss += self._train_batch(sentence_context, queries, expected_sentences, learning_rate)

            num_batches += 1

        return loss / (num_batches * self._batch_size) # Return average loss

    def _train_batch(self, sentence_context, queries, expected_sentences, learning_rate):
        # Prepare inputs, convert to numpy arrays
        feed_dict = {
            self._sentence_context: np.array(sentence_context),
            self._queries: np.array(queries),
            self._expected_sentences: np.array(expected_sentences),
            self._learning_rate: learning_rate
        }

        # Train
        loss, _ = self._sess.run([self._loss_op, self._train_op], feed_dict = feed_dict)
        return loss 

    def test(self, data):
        """ Tests the model on the given data without training it, returning the loss.

        @param data: (list(tuple(np.ndarray, np.ndarray, np.ndarray))) where the three numpy arrays represent a story, query, and 
            answer. Each story has shape (memory_size, max_sentence_length) and is a list of sentences, each query has shape (max_sentence_length,),
            and each answer has shape (vocab_size,). Each sentence is an array of words, where each word is represented by its one-hot index. The
            answer is also a word, but is expanded to its full one-hot vector form rather than just the index.

        @return float
        """
        # TODO Add functionality to support batch generators instead putting all data in memory

        # Extract data, prepare inputs
        sentence_context, queries, expected_sentences = zip(*data)
        feed_dict = {
            self._sentence_context: np.array(sentence_context),
            self._queries: np.array(queries),
            self._expected_sentences: np.array(expected_sentences)
        }

        # Compute model loss
        with self._sess.as_default():
            loss = self._sess.run(self._loss_op, feed_dict = feed_dict)

        return loss / len(data)
    
    def feedforward(self, sentence_context, queries):
        """ Given a set of sentences and a query, returns the index of the word representing the model's answer.
        
        @param sentence_context: (np.array) Numpy array with shape (memory_size, max_sentence_length) representing a list of
            context sentences.
        @param queries: (np.array) Numpy array with shape (max_sentence_length,) representing the query sentence.

        @return (np.array) Numpy array with shape (max_sentence_size,) representing the output sentence
        """
        # Prepare inputs and get predicted word
        with self._sess.as_default():
            feed_dict = {
                self._sentence_context: np.array([sentence_context]), # Batch size of 1
                self._queries: np.array([queries])
            }
            prediction = self._sess.run(self._predict_op, feed_dict = feed_dict)[0]
            prediction = np.argmax(prediction, axis = 1)

        return prediction

    def feedforward_raw(self, sentences):
        """ Given a list of the previous sentences, returns the model's predicted next sentence. Here all sentences are raw strings.
        
        @param sentences: (list(str)) List of raw sentences.
        
        @return str
        """
        # Split into context and query
        sentence_context, query = sentences[: -1], sentences[-1]

        # Tokenize
        sentence_context = [nltk.tokenize.word_tokenize(sentence) for sentence in sentences]
        query = nltk.tokenize.word_tokenize(query)

        # Map to one-hot indices
        sentence_context = [[self._word_to_index[word] for word in sentence] for sentence in sentence_context]
        query = [self._word_to_index[word] for word in query]

        # Get model prediction
        prediction = self.feedforward(sentence_context, queries)

        # Return raw sentence
        raw_prediction = " ".join([self._index_to_word[index] for index in prediction])
        return raw_prediction
    
    def save(self, index = None):
        """ Saves the model's current parameters in format (TF checkpoint) that can be recovered with the load function.

        @param index: (int) Index of the model to save, relative to other saved models.
        """
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
        """ Loads the most recent model or model with the given index from the directory of checkpoint models.
        
        @param index: (int) Index of model to load, or None to load model with highest index.
        """
        if index is None:
            index = len(os.listdir(self._checkpoint_dir)) - 1

        model_dir = "model_%i" % index
        model_file = os.path.join(self._checkpoint_dir, model_dir, self._name)

        saver = tf.train.Saver()
        saver.restore(self._sess, model_file)

    def close(self):
        """ Closes this model's tensorflow session.
        """
        self._sess.close()

# Utility functions

def position_encoding(sentence_size, embedding_dim):
    encoding = np.ones((embedding_dim, sentence_size), dtype = np.float32)
    ls = sentence_size + 1
    le = embedding_dim + 1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i - 1, j - 1] = (i - (embedding_dim + 1) / 2) * (j - (sentence_size + 1) / 2)
    encoding = 1 + 4 * encoding / embedding_dim / sentence_size 
    encoding[:, -1] = 1.0 # Make position encoding of time words identity to avoid modifying them

    return np.transpose(encoding)

def zero_nil_slot(t, name = None):
    with tf.name_scope(name, "zero_nil_slot", [t]) as name:
        t = tf.convert_to_tensor(t, name="t")
        s = tf.shape(t)[1]
        z = tf.zeros(tf.stack([1, s]))
        return tf.concat(axis=0, values=[z, tf.slice(t, [1, 0], [-1, -1])], name=name)

def add_gradient_noise(t, stddev = 1e-3, name = None):
    with tf.name_scope(name, "add_gradient_noise", [t, stddev]) as name:
        t = tf.convert_to_tensor(t, name = "t")
        gn = tf.random_normal(tf.shape(t), stddev = stddev)
        return tf.add(t, gn, name = name)
