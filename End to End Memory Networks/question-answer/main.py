from model import MNN
from data import Data
import tensorflow as tf
import time

def main():
    memory_size = 50
    data = Data("/home/ubuntu/data/babi_data/tasks_1-20_v1-2/en", memory_size)
    training, validation, testing = data.read_data()

    import os, shutil
    if os.path.exists("checkpoints") and os.path.isdir("checkpoints"):
        shutil.rmtree("checkpoints")

    mnn = MNN(data = data,
              batch_size = 32,
              memory_size = memory_size,
              embedding_dim = 40,
              nhops = 3,
              num_rnn_layers = 2,
              num_lstm_units = 200,
              lstm_forget_bias = 0.0,
              rnn_dropout_keep_prob = 0.5,
              init_lr = 0.01,
              max_grad_norm = 40)

    print("Training...")
    start = time.time()
    mnn.train(training, validation, verbose = True)
    end = time.time()
    print("Training took %s seconds" % str(end - start))

    mnn.save()
    mnn.close()

if __name__ == "__main__":
    main()
