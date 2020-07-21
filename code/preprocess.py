import numpy as np
import pickle

pre = '../data/cifar-10-batches-py/'


def load(file):
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
    return d


def trial():
    path = pre + 'data_batch_'
    for i in range(1, 6):
        whole = path + str(i)
        d = load(whole)
        print(len(d))
        for ele in d:
            print(ele)
        print(d[ele][:10])
        print(len(d[ele]))
        print(type(d[ele]))
    exit()


def build_training_set():
    print('Load data.')
    path = pre + 'data_batch_'
    dataset, label = [], []
    for i in range(1, 6):
        whole = path + str(i)
        dict = load(whole)
        label.extend(list(dict[b'labels']))
        dataset.extend(list(dict[b'data']))
    dataset = np.array(dataset)
    label = np.array(label)
    print(dataset.shape, label.shape)
    print('Loading success.')

    return dataset, label


if __name__ == '__main__':
    build_training_set()
