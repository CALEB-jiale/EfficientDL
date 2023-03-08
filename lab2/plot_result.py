import matplotlib.pyplot as plt
import numpy as np

PATH = "/Users/lucas/Documents/School/IMT/A2/S4/EfficientDL/lab2"
MODELS = ['dense_net_binary_cifar10_0001', 'dense_net_cifar10_0001']


def LoadData(model):

    PATH_train_acc = PATH + '/data_log/' + model + '_train_acc.npy'

    PATH_train_loss = PATH + '/data_log/' + model + '_train_loss.npy'

    PATH_train_time = PATH + '/data_log/' + model + '_train_time.npy'

    PATH_test_acc = PATH + '/data_log/' + model + '_test_acc.npy'

    PATH_test_loss = PATH + '/data_log/' + model + '_test_loss.npy'

    PATH_test_time = PATH + '/data_log/' + model + '_test_time.npy'

    train_acc = np.load(PATH_train_acc)
    train_loss = np.load(PATH_train_loss)
    train_time = np.load(PATH_train_time)
    test_acc = np.load(PATH_test_acc)
    test_loss = np.load(PATH_test_loss)
    test_time = np.load(PATH_test_time)

    return train_acc, train_loss, train_time, test_acc, test_loss, test_time


def PlotAccuracy(acc, mode, model):

    plt.figure()
    plt.plot(acc)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')
    title = mode + '_accuracy_for_' + model
    plt.title(title)
    plt.savefig(PATH + "/imgs/" + title + ".png")
    plt.show()


def PlotLoss(loss, mode, model):

    plt.figure()
    plt.plot(loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    title = mode + '_loss_for_' + model
    plt.title(title)
    plt.savefig(PATH + "/imgs/" + title + ".png")
    plt.show()


def PlotTime(time, mode, model):
    avg = sum(time)/len(time)

    plt.figure()
    plt.plot(time)
    plt.axhline(y=avg, color='r', linestyle='-.',
                label='Average = %.2f s' % avg)
    plt.xlabel('Epoch')
    plt.ylabel('Time/s')
    title = mode + '_time_for_' + model
    plt.title(title)
    plt.legend()
    plt.savefig(PATH + "/imgs/" + title + ".png")
    plt.show()


# model = MODELS[0]
# learning_rate = LearningRate[0]
# train_acc, train_loss, train_time, test_acc, test_loss, test_time = LoadData(
#     model, learning_rate)
# print(np.mean(train_time))

for model in MODELS:
    train_acc, train_loss, train_time, test_acc, test_loss, test_time = LoadData(
        model)
    # PlotAccuracy(train_acc, 'train', model)
    # PlotAccuracy(test_acc, 'test', model)
    # PlotLoss(train_loss, 'train', model)
    # PlotLoss(test_loss, 'test', model)
    # PlotTime(train_time, 'train', model)
    # PlotTime(test_time, 'test', model)
    print(test_acc[-1])
