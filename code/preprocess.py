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


def build_dataset():
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


def split_ds(dataset, label):
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=0, shuffle=True)
    print(x_train.shape, x_test.shape)
    return x_train, x_test, y_train, y_test


def visual_one_pic(pic):
    lst = [[[] for _ in range(32)] for _ in range(32)]
    idx = 0
    for c in range(3):
        for i in range(32):
            for j in range(32):
                lst[i][j].append(pic[idx])
                idx += 1
    new = np.array(lst)
    import cv2 as cv
    new = cv.cvtColor(new, cv.COLOR_RGB2BGR)
    new = cv.resize(new, (256, 256))
    cv.imshow('pic', new)
    cv.waitKey(0)


def dim_convert_1to3(data, show=False):
    l1 = data[:, :1024].reshape((data.shape[0], 32, 32, 1))
    l2 = data[:, 1024:2048].reshape((data.shape[0], 32, 32, 1))
    l3 = data[:, 2048:].reshape((data.shape[0], 32, 32, 1))
    result = np.concatenate((l1, l2, l3), axis=3)
    if not show:
        return result
    new = result[0]
    import cv2 as cv
    new = cv.cvtColor(new, cv.COLOR_RGB2BGR)
    new = cv.resize(new, (256, 256))
    cv.imshow('pic', new)
    cv.waitKey(0)


if __name__ == '__main__':
    dataset, label = build_dataset()
    # visual_one_pic(dataset[0])
    # exit()
    dim_convert_1to3(dataset, show=True)
    exit()

    x_train, x_test, y_train, y_test = split_ds(dataset, label)
