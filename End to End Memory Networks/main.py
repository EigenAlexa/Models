from model import MNN
from data import Data

def main():
    data = Data("/home/ubuntu/gutenberg_test")
    data_dir, data_name = "/home/ubuntu/converted_test_data", "gutenberg"
    data.convert(data_dir, data_name)

    mnn = MNN({"nhop": 1, "nepoch": 1})
    mnn.train(data_dir, data_name)
    print(mnn.test())

    mnn.save()
    mnn.close()

if __name__ == "__main__":
    main()
