from model import MNN
from data import Data

def main():
    data = Data("10.0.1.185")
    data_dir, data_name = "test_wiki", "wiki"
    data.convert((data_dir, data_name), collections = {"corpora": ["wiki"]})

    mnn = MNN({"nhop": 1, "nepoch": 1})
    mnn.train(data_dir, data_name)
    print(mnn.test())

    mnn.save()
    mnn.close()

if __name__ == "__main__":
    main()
