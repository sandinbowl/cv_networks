import torch
from preprocess import build_dataset, split_ds


def train_helper(model, save_name='lenet'):
    path = '../data/models_output/models_' + save_name
    print('Model saving path:', path)

    dataset, label = build_dataset()
    x_train, x_test, y_train, y_test = split_ds(dataset, label)

    x = torch.from_numpy(x_train)
    y = torch.from_numpy(y_train)
    if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()



if __name__ == '__main__':
    train_helper(0)

