from model import MNN
from data import Data
import time

def main():
    ut, ms, bs = 25, 50, 1
    data = Data(data_path = "/home/ubuntu/data/wikipedia",
                metadata_path = "/home/ubuntu/data/wikipedia/metadata",
                unk_threshold = ut,
                memory_size = ms,
                max_sentence_size = 120,
                batch_size = bs)

    print("Initializing...")
    mnn = MNN(data = data,
              batch_size = bs,
              memory_size = ms,
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
