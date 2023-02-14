import matplotlib.pyplot as plt
import numpy as np

PATH = "/Users/lucas/Documents/School/IMT/A2/S4/EfficientDL/lab1"
MODELS = ['VGG11', 'VGG13', 'VGG16', 'VGG19']
LearningRate = [0.01, 0.001]


def LoadData(model, learning_rate):
    lebel = ''
    if learning_rate == 0.01:
        label = '001'
    else:
        label = '0001'

    PATH_train_acc = PATH + '/data_log/' + model + \
        '_cifar_net_train_acc_' + label + '.npy'

    PATH_train_loss = PATH + '/data_log/' + model + \
        '_cifar_net_train_loss_' + label + '.npy'

    PATH_train_time = PATH + '/data_log/' + model + \
        '_cifar_net_train_time_' + label + '.npy'

    PATH_test_acc = PATH + '/data_log/' + model + \
        '_cifar_net_test_acc_' + label + '.npy'

    PATH_test_loss = PATH + '/data_log/' + model + \
        '_cifar_net_test_loss_' + label + '.npy'

    PATH_test_time = PATH + '/data_log/' + model + \
        '_cifar_net_test_time_' + label + '.npy'

    train_acc = np.load(PATH_train_acc)
    train_loss = np.load(PATH_train_loss)
    train_time = np.load(PATH_train_time)
    test_acc = np.load(PATH_test_acc)
    test_loss = np.load(PATH_test_loss)
    test_time = np.load(PATH_test_time)

    return train_acc, train_loss, train_time, test_acc, test_loss, test_time


def PlotAccuracy(acc, mode, model, lr):
    lebel = ''
    if lr == 0.01:
        label = '001'
    else:
        label = '0001'

    plt.figure()
    plt.plot(acc)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')
    title = mode + '_accuracy_for_' + model + '_' + label
    plt.title(title)
    plt.savefig(PATH + "/imgs/" + title + ".png")
    plt.show()


def PlotLoss(loss, mode, model, lr):
    lebel = ''
    if lr == 0.01:
        label = '001'
    else:
        label = '0001'

    plt.figure()
    plt.plot(loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    title = mode + '_loss_for_' + model + '_' + label
    plt.title(title)
    plt.savefig(PATH + "/imgs/" + title + ".png")
    plt.show()


def PlotAccuracyByModel(model, lr):
    lebel = ''
    if lr == 0.01:
        label = '001'
    else:
        label = '0001'

    train_acc, train_loss, train_time, test_acc, test_loss, test_time = LoadData(
        model, lr)

    plt.figure()
    plt.plot(train_acc, label=model + ' lr=' + str(lr) + ' train acc')
    plt.plot(test_acc, label=model + ' lr=' + str(lr) + ' test acc')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')
    title = 'train_test_accuracy_for_' + model + '_' + label
    plt.title(title)
    plt.savefig(PATH + "/imgs/" + title + ".png")
    plt.show()


def PlotLossByModel(model, lr):
    lebel = ''
    if lr == 0.01:
        label = '001'
    else:
        label = '0001'

    train_acc, train_loss, train_time, test_acc, test_loss, test_time = LoadData(
        model, lr)

    plt.figure()
    plt.plot(train_loss, label=model + ' lr=' + str(lr) + ' train loss')
    plt.plot(test_loss, label=model + ' lr=' + str(lr) + ' test loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    title = 'train_test_loss_for_' + model + '_' + label
    plt.title(title)
    plt.savefig(PATH + "/imgs/" + title + ".png")
    plt.show()


def PlotByModel():
    for model in MODELS:
        for learning_rate in LearningRate:
            train_acc, train_loss, train_time, test_acc, test_loss, test_time = LoadData(
                model, learning_rate)
            PlotAccuracy(train_acc, 'train', model, learning_rate)
            PlotLoss(train_loss, 'train', model, learning_rate)
            PlotAccuracy(test_acc, 'test', model, learning_rate)
            PlotLoss(test_loss, 'test', model, learning_rate)
            PlotAccuracyByModel(model, learning_rate)
            PlotLossByModel(model, learning_rate)


def PlotTrainAcc():
    plt.figure()

    for model in MODELS:
        for learning_rate in LearningRate:
            train_acc, train_loss, train_time, test_acc, test_loss, test_time = LoadData(
                model, learning_rate)
            plt.plot(train_acc, label=model + ' lr=' + str(learning_rate))

    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    title = 'train_accuracy'
    plt.title(title)
    plt.savefig(PATH + "/imgs/" + title + ".png")
    plt.show()


def PlotTrainLoss():
    plt.figure()

    for model in MODELS:
        for learning_rate in LearningRate:
            train_acc, train_loss, train_time, test_acc, test_loss, test_time = LoadData(
                model, learning_rate)
            plt.plot(train_loss, label=model + ' lr=' + str(learning_rate))

    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    title = 'train_loss'
    plt.title(title)
    plt.savefig(PATH + "/imgs/" + title + ".png")
    plt.show()


def PlotTestAcc():
    plt.figure()

    for model in MODELS:
        for learning_rate in LearningRate:
            train_acc, train_loss, train_time, test_acc, test_loss, test_time = LoadData(
                model, learning_rate)
            plt.plot(test_acc, label=model + ' lr=' + str(learning_rate))

    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    title = 'test_accuracy'
    plt.title(title)
    plt.savefig(PATH + "/imgs/" + title + ".png")
    plt.show()


def PlotTestLoss():
    plt.figure()

    for model in MODELS:
        for learning_rate in LearningRate:
            train_acc, train_loss, train_time, test_acc, test_loss, test_time = LoadData(
                model, learning_rate)
            plt.plot(test_loss, label=model + ' lr=' + str(learning_rate))

    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    title = 'test_loss'
    plt.title(title)
    plt.savefig(PATH + "/imgs/" + title + ".png")
    plt.show()


def PlotByMode():
    PlotTrainAcc()
    PlotTrainLoss()
    PlotTestAcc()
    PlotTestLoss()


def PlotTrainTimeVSTestAcc():
    plt.figure()
    for model in MODELS:
        for learning_rate in LearningRate:
            train_acc, train_loss, train_time, test_acc, test_loss, test_time = LoadData(
                model, learning_rate)
            plt.plot(sum(train_time), test_acc[-1], 'ro')
            plt.annotate(model + ' lr=' + str(learning_rate),
                         (sum(train_time), test_acc[-1]))

    plt.xlabel('Train Time(s)')
    plt.ylabel('Test Acc(%)')
    title = 'Train Time VS Test Acc'
    plt.title(title)
    plt.savefig(PATH + "/imgs/" + title + ".png")
    plt.show()


PlotByModel()
PlotByMode()
PlotTrainTimeVSTestAcc()

# model = MODELS[0]
# learning_rate = LearningRate[0]
# train_acc, train_loss, train_time, test_acc, test_loss, test_time = LoadData(
#     model, learning_rate)
# print(np.mean(train_time))
