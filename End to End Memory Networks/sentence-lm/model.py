import os, math, random, sys, types
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
        self._data = data
        self._sentence_size = self._num_rnn_steps = data.max_sentence_size
        self._vocab_size = len(data.word_to_index)

        # Hyperparameters
        self._batch_size = batch_size
        self._memory_size = memory_size
        self._embedding_dim = embedding_dim
        self._hops = nhops
        self._num_rnn_layers = num_rnn_layers
        self._num_lstm_units = num_lstm_units
        self._lstm_forget_bias = lstm_forget_bias
        self._rnn_dropout_keep_prob = rnn_dropout_keep_prob
        self._init_lr = init_lr
        self._max_grad_norm = max_grad_norm
        self._nonlin = nonlin
        self._initializer = initializer

        self._sess = sess
        self._name = name
        self._checkpoint_dir = "./checkpoints" 
        self._encoding = tf.constant(position_encoding(self._sentence_size, self._embedding_dim), name = "encoding") # Sentence encoding vector

        # Set up the model architecture
        self._build_inputs()
        self._build_params()
        self._build_model()
        self._build_training()

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
            null_word_slot = tf.zeros([1, self._embedding_dim])
            A = tf.concat(axis = 0, values = [null_word_slot, self._initializer([self._vocab_size - 1, self._embedding_dim])])
            C = tf.concat(axis = 0, values = [null_word_slot, self._initializer([self._vocab_size - 1, self._embedding_dim])])

            # Adjacent weight sharing - each embedding in self.C is the output embedding for layer l and memory embedding for layer l + 1
            self.A_1 = tf.Variable(A, name = "A") # Initial memory embedding
            self.C = []

            # Build output embedding matrices
            for hopn in range(self._hops):
                with tf.variable_scope('hop_{}'.format(hopn)):
                    self.C.append(tf.Variable(C, name = "C"))

            # Answer prediction weight matrix
            self.W = tf.Variable(self._initializer([self._embedding_dim, self._embedding_dim]), name = "W")

            # Linear mapping for layer output (not necessary with adjacent weight sharing)
            # self.H = tf.Variable(self._initializer([self._embedding_dim, self._embedding_dim]), name = "H") 

        self._null_vars = set([self.A_1.name] + [x.name for x in self.C])

        # Build weights for output of RNN
        self._rnn_W = tf.Variable(self._initializer([self._num_lstm_units, self._vocab_size]), name = "rnn_W")
        self._rnn_b = tf.Variable(self._initializer([self._vocab_size]), name = "rnn_b")

    def _build_model(self):
        with tf.variable_scope(self._name):
            # Use A_1 for the query embedding as per Adjacent Weight Sharing
            q_emb = tf.nn.embedding_lookup(self.A_1, self._queries)
            u = sentence_representation(q_emb, self._encoding)

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
            # m_A = tf.reduce_sum(m_emb_A * self._encoding, axis = 2) # Convert to sentence representation
            m_A = sentence_representation(m_emb_A, self._encoding)
        else:
            with tf.variable_scope('hop_{}'.format(hopn - 1)):
                m_emb_A = tf.nn.embedding_lookup(self.C[hopn - 1], self._sentence_context) # (Adjacent weight sharing) Previous output embedding becomes memory embedding
                # m_A = tf.reduce_sum(m_emb_A * self._encoding, axis = 2) # Convert to sentence representation
                m_A = sentence_representation(m_emb_A, self._encoding)

        # Compute probability vector, by passing the cosine similarities between (encoded) query and (encoded) memory vectors
        u_temp = tf.transpose(tf.expand_dims(prev_u, -1), [0, 2, 1]) # Hack to get around no reduce_dot
        dotted = tf.reduce_sum(m_A * u_temp, 2)
        p = tf.nn.softmax(dotted)

        # Compute output vectors (context sentences encoded with output embedding)
        with tf.variable_scope('hop_{}'.format(hopn)):
            m_emb_C = tf.nn.embedding_lookup(self.C[hopn], self._sentence_context)
        # m_C = tf.reduce_sum(m_emb_C * self._encoding, axis = 2)
        m_C = sentence_representation(m_emb_C, self._encoding)

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

        # Add degenerate gradients null-paddings
        null_grads_and_vars = []
        for g, v in grads_and_vars:
            if v.name in self._null_vars:
                null_grads_and_vars.append((zero_null_slot(g), v))
            else:
                null_grads_and_vars.append((g, v))

        self._train_op = self._opt.apply_gradients(null_grads_and_vars, name = "train_op")

    def _loss_function(self):
        # Approach 1 (default sequence to sequence loss on one-hot words)
        # logits = differentiable_argmax(self._output, beta = 1)
        # loss = tf.contrib.seq2seq.sequence_loss(
                # logits = [logits],
                # targets = [self._expected_sentences],
                # weights = [tf.ones([self._batch_size])])
                # softmax_loss_function = lambda t, l: tf.nn.softmax_cross_entropy_with_logits(labels = t, logits = l))

        # Approach 2 (sum of cosine distances between embedded words)
        emb_output = embedding_with_floats(self.A_1, self._output, beta = 1) # Hack to embed non-integral one-hot vectors
        emb_expected = tf.nn.embedding_lookup(self.A_1, self._expected_sentences)
        distances = tf.reduce_sum(cosine_distance(emb_output, emb_expected), axis = -1)
        loss = tf.reduce_mean(distances) # Average over batches

        # Approach 3 (standard loss function between embedded sentence vector (ie with position encoding))
        # emb_sentence_output = sentence_representation(embedding_with_floats(self.A_1, self._output, beta = 1), self._encoding)
        # emb_sentence_expected = sentence_representation(tf.nn.embedding_lookup(self.A_1, self._expected_sentences), self._encoding)
        # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = emb_sentence_output, labels = emb_expected_sentences)
        # loss = tf.reduce_mean(cross_entropy) # Average over batches

        # TODO Approach 4 (sum of approaches (1 or 2) and approach 3)
        # emb_output = embedding_with_floats(self.A_1, self._output, beta = 1) # Hack to embed non-integral one-hot vectors
        # emb_expected = tf.nn.embedding_lookup(self.A_1, self._expected_sentences)
        # emb_sentence_output = sentence_representation(emb_output, self._encoding)
        # emb_sentence_expected = sentence_representation(emb_expected, self._encoding)

        # cosine_distance_loss = tf.reduce_mean(tf.reduce_sum(cosine_distance(emb_output, emb_expected)))
        # cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = emb_sentence_output, labels = emb_sentence_expected))
        # loss = cosine_distance_loss + cross_entropy_loss

        # TODO Approach 5 (train a separate ML model (NOT jointly with this model) to act as a good loss function specifically for NLP, capturing semantic meaning behind sentences)

        # TODO Approach 6 (sum of approaches 4 and 5)

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
        if verbose:
            print_hyperparameters(self)

        with self._sess.as_default():
            learning_rate = self._init_lr
            old_valid_loss = float("inf")

            # Train until convergence or nepochs have passed
            annealed_count = epoch = 0
            while (nepochs == -1 and annealed_count <= 5) or (nepochs != -1 and epoch < nepochs):
                if verbose:
                    print("Epoch %i" % epoch)

                # Train epoch
                train_loss = self._train_batches_SGD(train_data, learning_rate, verbose)
                valid_loss = self.test(valid_data)

                if verbose:
                    clear_line()
                    print("\tTraining loss: %s" % str(train_loss))
                    print("\tValidation loss: %s" % str(valid_loss))
                    print("\tLearning rate: %s" % str(learning_rate))
                    print()

                # Learning rate annealing
                if valid_loss > old_valid_loss * 0.9999:
                    learning_rate *= 2 / 3
                    annealed_count += 1
                else:
                    annealed_count = 0

                old_valid_loss = valid_loss
                epoch += 1

        return (train_loss, valid_loss)

    def _train_batches_SGD(self, data, learning_rate, verbose = True):
        loss, num_batches = 0.0, 0

        # Prepare batches
        if type(data) == list:
            batches = [data[i : i + self._batch_size] for i in range(0, len(data) - self._batch_size, self._batch_size)]
            random.shuffle(batches)
        else: # data is a generator that yields batches, to save memory
            batches = data

        # Train each batch
        for batch in batches:
            # if num_batches % 20 == 0:
            if False:
                test_files = os.listdir("/home/ubuntu/data/wikipedia/test")
                sentences = []
                # while len(sentences) < 3:
                while len(sentences) < self._memory_size + 2:
                    with open(os.path.join("/home/ubuntu/data/wikipedia/test", test_files[int(random.random() * len(test_files))]), "r") as f:
                        sentences = f.readlines()
                        if len(sentences) >= self._memory_size + 2:
                            sentences = sentences[: self._memory_size + 2]
                predicted = self.feedforward_raw(sentences)
                # print("\nCONTEXT")
                # for sentence in sentences[: -1]:
                    # print(sentence)
                # print("\nEXPECTED: %s" % sentences[-1])
                # print("PREDICTED: %s" % predicted)
                # print()

            if verbose:
                clear_line()
                print("\tTraining batch %i" % num_batches,  end = "\r")

            # TODO parallelize training of each batch across all GPUs
            # Extract batch and train
            sentence_context, queries, expected_sentences = zip(*batch)
            feed_dict = {
                self._sentence_context: np.array(sentence_context),
                self._queries: np.array(queries),
                self._expected_sentences: np.array(expected_sentences),
                self._learning_rate: learning_rate
            }

            emb_output = embedding_with_floats(self.A_1, self._output, beta = 1) # Hack to embed non-integral one-hot vectors
            emb_expected = tf.nn.embedding_lookup(self.A_1, self._expected_sentences)

            cos_dists = cosine_distance(emb_output, emb_expected)
            x, y, z = self._sess.run([emb_output, emb_expected, cos_dists], feed_dict)
            print(x.shape, y.shape, z.shape)
            # print("Cosine distances")
            # print(self._sess.run(cos_dists, feed_dict))
            # print()
            # distances = tf.reduce_sum(cos_dists, axis = -1)
            # print("Distances")
            # print(self._sess.run(distances, feed_dict))
            # print()
            # loss = tf.reduce_mean(distances) # Average over batches
            # print("Loss")
            # print(self._sess.run(loss, feed_dict))
            __import__("sys").exit()

            batch_loss, _ = self._sess.run([self._loss_op, self._train_op], feed_dict = feed_dict)

            # for var in tf.trainable_variables():
                # print(var)
                # print(self._sess.run(var))
                # print()
            # print(self._sess.run(self._rnn_W))
            # print(self._sess.run(self._rnn_W))
            # print(self._sess.run(self._rnn_b))
            # print(self._sess.run(self.W))
            __import__("sys").exit()

            loss += batch_loss
            num_batches += 1

        return loss / (num_batches * self._batch_size) # Return average loss

    def test(self, data):
        """ Tests the model on the given data without training it, returning the loss.

        @param data: (list(tuple(np.ndarray, np.ndarray, np.ndarray))) where the three numpy arrays represent a story, query, and 
            answer. Each story has shape (memory_size, max_sentence_length) and is a list of sentences, each query has shape (max_sentence_length,),
            and each answer has shape (vocab_size,). Each sentence is an array of words, where each word is represented by its one-hot index. The
            answer is also a word, but is expanded to its full one-hot vector form rather than just the index.

        @return float
        """
        if isinstance(data, types.GeneratorType):
            data = list(data)

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
            prediction = self._sess.run(self._output, feed_dict = feed_dict)[0]
            prediction = np.argmax(prediction, axis = 1)

        return prediction

    def feedforward_raw(self, sentences):
        """ Given a list of the previous sentences, returns the model's predicted next sentence. Here all sentences are raw strings.
        
        @param sentences: (list(str)) List of raw sentences.
        
        @return str
        """
        # Convert to indices and remove sentences that are too long
        sentences = [self._data.process_raw(sentence) for sentence in sentences]
        sentences = [sentence for sentence in sentences if len(sentence) <= self._sentence_size]

        # Split into context and query, fitting sentence context into memory
        sentence_context, query = sentences[: -1], sentences[-1]
        if len(sentence_context) > self._memory_size:
            sentence_context = sentence_context[- self._memory_size :]
        elif len(sentence_context) < self._memory_size:
            sentence_context = [[0 for _ in range(self._sentence_size)] for _ in range(self._memory_size - len(sentence_context))] + sentence_context

        # Get model prediction
        prediction = self.feedforward(sentence_context, query)

        # Return raw sentence
        raw_prediction = self._data.unprocess(prediction)
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

def sentence_representation(embedded_sentences, encoding, axis = -2):
    return tf.reduce_sum(embedded_sentences * encoding, axis = axis)

def zero_null_slot(t, name = None):
    with tf.name_scope(name, "zero_null_slot", [t]) as name:
        t = tf.convert_to_tensor(t, name="t")
        s = tf.shape(t)[1]
        z = tf.zeros(tf.stack([1, s]))
        return tf.concat(axis=0, values=[z, tf.slice(t, [1, 0], [-1, -1])], name=name)

def add_gradient_noise(t, stddev = 1e-3, name = None):
    with tf.name_scope(name, "add_gradient_noise", [t, stddev]) as name:
        t = tf.convert_to_tensor(t, name = "t")
        gn = tf.random_normal(tf.shape(t), stddev = stddev)
        return tf.add(t, gn, name = name)

def embedding_with_floats(embedding_matrix, x, beta = 1, axis = -1):
    # Accentuate existing (unnormalized) probability distribution on x
    x = tf.nn.softmax(beta * x)

    embeddings = tf.einsum("ijk,kl->ijl", x, embedding_matrix)
    return embeddings

def clear_line():
    sys.stdout.write("\033[K") # Clear to the end of line

def dot_product(x, y):
    return tf.reduce_sum(tf.multiply(x, y), axis = -1)

def magnitude(x):
    return tf.sqrt(dot_product(x, x))

def cosine_distance(x, y, epsilon = 0.00001):
    inner_prod = dot_product(x, y)
    # mag_prod = tf.multiply(magnitude(x) + epsilon, magnitude(y) + epsilon) # Prevent divide by zero
    mag_prod = tf.multiply(magnitude(x), magnitude(y)) + epsilon # Prevent divide by zero

    similarity = tf.divide(inner_prod, mag_prod)
    return (1 - similarity) / 2 # Negate and map to [0, 1]

def print_hyperparameters(model):
    print("Hyperparameters:")
    print("\tMaximum sentence size: %s" % str(model._sentence_size))
    print("\tVocabulary size: %s" % str(model._vocab_size))
    print("\tBatch size: %s" % str(model._batch_size))
    print("\tMemory size: %s" % str(model._memory_size))
    print("\tEmbedding dimension: %s" % str(model._embedding_dim))
    print("\tNumber of hops: %s" % str(model._hops))
    print("\tNumber of RNN layers: %s" % str(model._num_rnn_layers))
    print("\tNumber of LSTM units per RNN layer: %s" % str(model._num_lstm_units))
    print("\tLSTM forget bias: %s" % str(model._lstm_forget_bias))
    print("\tRNN dropout keep probability: %s" % str(model._rnn_dropout_keep_prob))
    print("\tInitial learning rate: %s" % str(model._init_lr))
    print("\tMaximum gradient norm: %s" % str(model._max_grad_norm))
    print("\tNon-linearity: %s" % str(model._nonlin))
    print()
