from model import MNN
from data import Data

import tensorflow as tf

def main():
    memory_size = 50
    data = Data("/home/ubuntu/data/babi_data/tasks_1-20_v1-2/en", memory_size)
    training, validation, testing = data.read_data()

    import os, shutil
    if os.path.exists("checkpoints") and os.path.isdir("checkpoints"):
        shutil.rmtree("checkpoints")

    mnn = MNN(data = data,
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
    # prediction = mnn.feed(testing[0][0], testing[0][1])
    # print(prediction)

    # mnn.save()
    mnn.close()

if __name__ == "__main__":
    main()
