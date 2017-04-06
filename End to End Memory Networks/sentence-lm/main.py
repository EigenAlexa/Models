from model import MNN
from data import Data
import tensorflow as tf
import time

def main():
    memory_size, batch_size = 50, 32
    data = Data("/home/ubuntu/data/wikipedia", "/home/ubuntu/data/wikipedia/metadata", memory_size, batch_size)

    # TODO delete
    import os, shutil
    if os.path.exists("checkpoints") and os.path.isdir("checkpoints"):
        shutil.rmtree("checkpoints")

    mnn = MNN(data = data,
              batch_size = batch_size,
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
    mnn.train(data.get_batches("/home/ubuntu/data/wikipedia/train"), data.get_batches("/home/ubuntu/data/wikipedia/validation"))
    end = time.time()
    print("Training took %s seconds" % str(end - start))

    mnn.save()
    mnn.close()

if __name__ == "__main__":
    main()
