from model import MNN
from data import Data

def main():
    # data = Data("10.0.1.185")
    # data_dir, data_name = "test_wiki", "wiki"
    # data.convert((data_dir, data_name), collections = {"corpora": ["wiki"]})
    # data = Data(source = (data_dir, data_name))
    # train_data, valid_data, test_data, word2idx = data.read_data()

    data = Data(source = ("/home/ubuntu/data/ptb_data","ptb"), raw = False)
    (train_data, valid_data, test_data), word2idx = data.read_data()
    idx2word = dict(zip(word2idx.values(), word2idx.keys()))

    mnn = MNN(len(word2idx), metadata = {"nwords": len(word2idx), "nhop": 6, "nepoch": 100, "show": True})

    print("Starting to train")
    mnn.train(train_data, valid_data)

    print(mnn.test())

    mnn.save()
    mnn.close()

if __name__ == "__main__":
    main()
