from model import MNN
from data import Data

def main():
    # data = Data("10.0.1.185")
    # data_dir, data_name = "test_wiki", "wiki"
    # data.convert((data_dir, data_name), collections = {"corpora": ["wiki"]})
    # data = Data(source = (data_dir, data_name))
    # train_data, valid_data, test_data, word2idx = data.read_data()

    memory_size = 50
    data = Data("/home/ubuntu/memn2n/data/tasks_1-20_v1-2/en", memory_size)
    training, validation, testing = data.read_data()

    mnn = MNN(vocab_size = data.vocab_size(),
              max_sent_size = data.max_sent_size(),
              batch_size = 32,
              mem_size = memory_size,
              emb_dim = 40,
              nhops = 3,
              nepochs = 100,
              max_grad_norm = 40,
              init_lr = 0.01,
              init_hid = 0.1,
              init_std = 0.1)
    mnn.train(training, validation, verbose = True)

    # mnn.save()
    # mnn.close()

if __name__ == "__main__":
    main()
