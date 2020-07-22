# Decision Board:
# https://docs.qq.com/doc/DTnlicHFFUVlyaFBL
import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
import time
from preprocess import build_dataset, split_ds, dim_convert_1to3


def train_helper(model, lr=0.5, bs=64, max_iter=100, save_name='lenet'):
    """
    # Save and reload model with state_dict. State_dict will save plenty of memory space.
    # https://blog.csdn.net/Iam_Human/article/details/102763851
    PATH = './model.pt'
    torch.save(model.state_dict(), PATH)
    model2 = TheModelClass()
    model2.load_state_dict(torch.load(PATH))
    model2.eval()

    # Official tutorial for the main training precess can be found at:
    # https://pytorch.org/tutorials/beginner/nn_tutorial.html
    """
    path = '../data/models_output/models_' + save_name
    print('Model saving path:', path)

    dataset, label = build_dataset()
    dataset = dataset.reshape((dataset.shape[0], 3, 32, 32))
    x_train, x_test, y_train, y_test = split_ds(dataset, label)

    # Loss function and optimization function.
    loss_function = F.cross_entropy
    opt = optim.SGD(model.parameters(), lr=lr)

    # Convert training data into torch.tensor type and try to use cuda to improve the training speed.
    x = torch.from_numpy(x_train).float()
    y = torch.from_numpy(y_train).long()
    if torch.cuda.is_available():
        x, y = x.cuda(), y.cuda()
        model = model.cuda()
    deal_dataset = TensorDataset(Variable(x), Variable(y))
    train_dl = DataLoader(dataset=deal_dataset, batch_size=bs)

    # Deal with validation dataset in the same way
    x = torch.from_numpy(x_test).float()
    # x = torch.from_numpy(x_test).view(x_shape[0], 1, x_shape[1])
    y = torch.from_numpy(y_test).long()
    if torch.cuda.is_available(): x, y = x.cuda(), y.cuda()
    deal_dataset = TensorDataset(Variable(x), Variable(y))
    valid_dl = DataLoader(dataset=deal_dataset, batch_size=bs)

    # Train, save model at each epoch, evaluate.
    print('Start training.')
    print("Current Time:", time.asctime(time.localtime(time.time())))
    best_model = -1
    least_loss, least_acc = -1, -1
    for epoch in range(max_iter):
        model.train()
        for xb, yb in train_dl:
            loss = loss_function(model(xb), yb)
            loss.backward()
            opt.step()
            opt.zero_grad()
        # PATH = path + '/m_' + save_name + str(epoch) + '.pt'
        # torch.save(model.state_dict(), PATH)
        model.eval()
        with torch.no_grad():
            valid_loss = sum(loss_function(model(xb), yb) for xb, yb in valid_dl) / len(valid_dl)
            valid_acc = sum(torch.eq(model(xb).argmax(dim=1), yb).sum().float().item() for xb, yb in valid_dl) / float(len(y_test)) * 100
        if least_loss == -1 or valid_loss < least_loss:
            least_loss, least_acc = valid_loss, valid_acc
            best_model = epoch
        # print('At epoch', epoch, 'validation loss:', valid_loss)
        print('At epoch', epoch, 'validation accuracy (%):', valid_acc)

        print("Current Time:", time.asctime(time.localtime(time.time())))

    print("Success")
    print('Best model: model', best_model, ', acc:', least_acc)


if __name__ == '__main__':
    from lenet import LeNet
    model = LeNet()
    train_helper(model)

