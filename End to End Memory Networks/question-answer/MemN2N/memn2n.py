"""End-To-End Memory Networks.

The implementation is based on http://arxiv.org/abs/1503.08895 [1]
"""
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
from six.moves import range
from sklearn import metrics
import random, math

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
    with tf.op_scope([t], name, "zero_nil_slot") as name:
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
    with tf.op_scope([t, stddev], name, "add_gradient_noise") as name:
        t = tf.convert_to_tensor(t, name = "t")
        gn = tf.random_normal(tf.shape(t), stddev = stddev)
        return tf.add(t, gn, name = name)

class MemN2N(object):
    """End-To-End Memory Network."""
    def __init__(self, batch_size, vocab_size, sentence_size, memory_size, embedding_size,
        hops = 3,
        init_lr = 0.01,
        max_grad_norm = 40.0,
        nonlin = None,
        initializer = tf.random_normal_initializer(stddev = 0.1),
        encoding = position_encoding,
        session = tf.Session(),
        name = "MemN2N"):
        """Creates an End-To-End Memory Network

        Args:
            batch_size: The size of the batch.

            vocab_size: The size of the vocabulary (should include the nil word). The nil word
            one-hot encoding should be 0.

            sentence_size: The max size of a sentence in the data. All sentences should be padded
            to this length. If padding is required it should be done with nil one-hot encoding (0).

            memory_size: The max size of the memory. Since Tensorflow currently does not support jagged arrays
            all memories must be padded to this length. If padding is required, the extra memories should be
            empty memories; memories filled with the nil word ([0, 0, 0, ......, 0]).

            embedding_size: The size of the word embedding.

            hops: The number of hops. A hop consists of reading and addressing a memory slot.
            Defaults to `3`.

            max_grad_norm: Maximum L2 norm clipping value. Defaults to `40.0`.

            nonlin: Non-linearity. Defaults to `None`.

            initializer: Weight initializer. Defaults to `tf.random_normal_initializer(stddev=0.1)`.

            optimizer: Optimizer algorithm used for SGD. Defaults to `tf.train.AdamOptimizer(learning_rate=1e-2)`.

            encoding: A function returning a 2D Tensor (sentence_size, embedding_size). Defaults to `position_encoding`.

            session: Tensorflow Session the model is run with. Defaults to `tf.Session()`.

            name: Name of the End-To-End Memory Network. Defaults to `MemN2N`.
        """
        self._batch_size = batch_size
        self._vocab_size = vocab_size
        self._sentence_size = sentence_size
        self._memory_size = memory_size
        self._embedding_size = embedding_size
        self._hops = hops
        self._init_lr = init_lr
        self._max_grad_norm = max_grad_norm
        self._nonlin = nonlin
        self._init = initializer
        self._name = name

        self._encoding = tf.constant(encoding(self._sentence_size, self._embedding_size), name = "encoding") # Encoding vector

        # Set up the model architecture
        self._build_inputs()
        self._build_params()

        # Cross entropy loss
        logits = self._feedforward(self._sentence_context, self._queries) # (batch_size, vocab_size)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = tf.cast(self._answers, tf.float32), name = "cross_entropy")
        cross_entropy_sum = tf.reduce_sum(cross_entropy, name = "cross_entropy_sum")
        loss_op = cross_entropy_sum

        # Gradient pipeline
        self._opt = tf.train.GradientDescentOptimizer(learning_rate = self._init_lr)
        grads_and_vars = self._opt.compute_gradients(loss_op)
        grads_and_vars = [(tf.clip_by_norm(g, self._max_grad_norm), v) for g,v in grads_and_vars]
        grads_and_vars = [(add_gradient_noise(g), v) for g, v in grads_and_vars]
        nil_grads_and_vars = []
        for g, v in grads_and_vars:
            if v.name in self._nil_vars:
                nil_grads_and_vars.append((zero_nil_slot(g), v))
            else:
                nil_grads_and_vars.append((g, v))
        train_op = self._opt.apply_gradients(nil_grads_and_vars, name = "train_op")

        # Predict ops
        predict_op = tf.argmax(logits, 1, name = "predict_op") # Model answer is one-hot vector
        predict_proba_op = tf.nn.softmax(logits, name = "predict_proba_op")
        predict_log_proba_op = tf.log(predict_proba_op, name = "predict_log_proba_op")

        # Assign ops
        self.loss_op = loss_op
        self.predict_op = predict_op
        self.predict_proba_op = predict_proba_op
        self.predict_log_proba_op = predict_log_proba_op
        self.train_op = train_op

        init_op = tf.global_variables_initializer()
        self._sess = session
        self._sess.run(init_op)

    def _build_inputs(self):
        # Create nodes for expected inputs
        self._sentence_context = tf.placeholder(tf.int32, [None, self._memory_size, self._sentence_size], name = "sentence_context") # Memory matrix
        self._queries = tf.placeholder(tf.int32, [None, self._sentence_size], name = "queries") # Query sentence
        self._answers = tf.placeholder(tf.int32, [None, self._vocab_size], name = "answers") # Answer

    def _build_params(self):
        # Create nodes for model parameters, which are the embedding matrices
        with tf.variable_scope(self._name):
            nil_word_slot = tf.zeros([1, self._embedding_size])
            A = tf.concat(axis = 0, values = [nil_word_slot, self._init([self._vocab_size - 1, self._embedding_size])])
            C = tf.concat(axis = 0, values = [nil_word_slot, self._init([self._vocab_size - 1, self._embedding_size])])

            # Adjacent weight sharing - each embedding in self.C is the output embedding for layer l and memory embedding for layer l + 1
            self.A_1 = tf.Variable(A, name = "A") # Initial memory embedding
            self.C = []

            for hopn in range(self._hops):
                with tf.variable_scope('hop_{}'.format(hopn)):
                    self.C.append(tf.Variable(C, name = "C"))

            # self.C[-1] will act as the answer prediction weight matrix
            # self.W = tf.Variable(self._init([self._embedding_size, self._vocab_size]), name = "W")

            # Linear mapping for layer output (not necessary with adjacent weight sharing)
            # self.H = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name = "H") 

        self._nil_vars = set([self.A_1.name] + [x.name for x in self.C])

    def _feedforward(self, sentence_context, queries):
        # Feedforward input through the model
        with tf.variable_scope(self._name):
            # Use A_1 for the query embedding as per Adjacent Weight Sharing
            q_emb = tf.nn.embedding_lookup(self.A_1, queries)
            u_0 = tf.reduce_sum(q_emb * self._encoding, axis = 1)
            u = [u_0]

            for hopn in range(self._hops):
                # Store context sentences in memory, encoded with memory embedding
                if hopn == 0:
                    m_emb_A = tf.nn.embedding_lookup(self.A_1, sentence_context) # Get word embeddings
                    m_A = tf.reduce_sum(m_emb_A * self._encoding, 2) # Convert to sentence representation

                else:
                    with tf.variable_scope('hop_{}'.format(hopn - 1)):
                        m_emb_A = tf.nn.embedding_lookup(self.C[hopn - 1], sentence_context) # Last output embedding becomes memory embedding
                        m_A = tf.reduce_sum(m_emb_A * self._encoding, 2) # Convert to sentence representation

                # Compute probability vector, by passing the cosine similarities between (encoded) query and (encoded) memory vectors
                u_temp = tf.transpose(tf.expand_dims(u[-1], -1), [0, 2, 1]) # Hack to get around no reduce_dot
                dotted = tf.reduce_sum(m_A * u_temp, 2)
                p = tf.nn.softmax(dotted)

                # Compute output vectors (context sentences encoded with output embedding)
                with tf.variable_scope('hop_{}'.format(hopn)):
                    m_emb_C = tf.nn.embedding_lookup(self.C[hopn], sentence_context)
                m_C = tf.reduce_sum(m_emb_C * self._encoding, 2)

                p_temp = tf.transpose(tf.expand_dims(p, -1), [0, 2, 1])
                c_temp = tf.transpose(m_C, [0, 2, 1])
                o_k = tf.reduce_sum(c_temp * p_temp, 2) # Response vector

                u_k = u[-1] + o_k # Layer output
                # u_k = tf.matmul(u[-1], self.H) + o_k # Layer output with linear mapping (not necessary with adjacent weight sharing)

                # Nonlinearity
                if self._nonlin:
                    u_k = nonlin(u_k)

                u.append(u_k)

            # Use last C for output (transposed)
            with tf.variable_scope('hop_{}'.format(self._hops)):
                return tf.matmul(u_k, tf.transpose(self.C[-1]))

    def train(self, train_data, valid_data, nepochs, verbose = True):
        """ Trains the model on the given training and validation data.

        @param train_data: (list(tuple(np.ndarray, np.ndarray, np.ndarray))) where the three numpy arrays represent a story, query, and 
            answer. Each story has shape (memory_size, max_sentence_length) and is a list of sentences, each query has shape (max_sentence_length,),
            and each answer has shape (vocab_size,). Each sentence is an array of words, where each word is represented by its one-hot index. The
            answer is also a word, but is expanded to its full one-hot vector form rather than just the index.
        @param valid_data: (list(tuple(np.ndarray, np.ndarray, np.ndarray))) Same as training data, but used
            for validation.
        @param nepochs: (int) How many epochs to train for.
        """
        # anneal_stop_epoch, anneal_rate = nepochs, 25
        learning_rate = self._init_lr
        old_valid_loss = float("inf")

        for i in range(nepochs):
            # if i <= anneal_stop_epoch:
                # anneal = 2**(i // anneal_rate)
            # else:
                # anneal = 2**(anneal_stop_epoch // anneal_rate)

            # learning_rate = self._init_lr / anneal

            train_loss = self._train_batches_SGD(train_data, self._batch_size, learning_rate)
            valid_loss = self.test(valid_data)

            # Learning rate annealing
            if valid_loss > old_valid_loss * 0.9999:
                learning_rate *= 2 / 3

            if learning_rate < 0.000001:
                break

            if verbose and i % 10 == 0:
                valid_acc = self.accuracy(valid_data)
                print("Epoch %i" % i)
                print("\tTraining loss: %s" % str(train_loss))
                print("\tValidation loss: %s" % str(valid_loss))
                print("\tValidation accuracy: %s" % str(valid_acc))
                print()

            old_valid_loss = valid_loss

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
        feed_dict = {self._sentence_context: sentence_context, self._queries: queries, self._answers: answers}
        loss, _ = self._sess.run([self.loss_op, self.train_op], feed_dict = feed_dict)
        return loss 

    def test(self, data):
        """ Runs the model on the data and returns the loss, without training the model.
        """
        sentence_context, queries, answers = zip(*data)
        sentence_context, queries, answers = np.array(sentence_context), np.array(queries), np.array(answers)

        feed = {self._sentence_context: sentence_context, self._queries: queries, self._answers: answers}
        loss = self._sess.run(self.loss_op, feed_dict = feed)
        return loss / len(data)
    
    def accuracy(self, data):
        sentence_context, queries, answers = zip(*data)
        sentence_context, queries, answers = np.array(sentence_context), np.array(queries), np.array(answers)
        labels = np.argmax(answers, axis = 1)

        predictions = self.predict(sentence_context, queries)
        acc = metrics.accuracy_score(predictions, labels)

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
        return self._sess.run(self.predict_op, feed_dict = feed_dict)
    
    def save(self, name = None):
        if name is None:
            name = self._name

        saver = tf.train.Saver()
        saver.save(self._sess, name)

    def load(self, model_file = None):
        if model_file is None:
            model_file = self._name

        saver = tf.train.import_meta_graph(model_file)
        saver.restore(self._sess, tf.train.latest_checkpoint("./"))
