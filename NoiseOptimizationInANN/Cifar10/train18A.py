import os
import pickle as pkl

import torch.nn.init
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

import Resnet18

if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
else:
    pass

device = Resnet18.device

if __name__ == '__main__':

    resnet18_model = [Resnet18.getResNet18PlusA, Resnet18.getResNet18NA]

    resnet18_name = ['Resnet18.getResNet18PlusA', 'Resnet18.getResNet18NA']

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    batch_size = 128
    epoches = 50
    loss = 0.
    train_dataset = datasets.CIFAR10(root='../../LR/EBP', train=True, transform=transform_train, download=True)
    test_dataset = datasets.CIFAR10(root='../../LR/EBP', train=False, transform=transform_test, download=True)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    transform = transforms.Compose([transforms.ToTensor()])

    for j in range(len(resnet18_model)):
        print('new model training:{}'.format(resnet18_name[j]))
        model = resnet18_model[j]()
        model = model.to(device)
        trainLoss = 0.
        testLoss = 0.
        learning_rate = 1e-3
        start_epoch = 0
        test_loss_list = []
        train_loss_list = []
        acc_list = []
        epoches = 50
        SoftmaxWithXent = nn.CrossEntropyLoss()
        # define optimization algorithm
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-04)
        print('{} epoch to run:{} learning rate:{}'.format(resnet18_name[j], epoches, learning_rate))
        for epoch in range(start_epoch, start_epoch + epoches):
            train_N = 0.
            train_n = 0.
            trainLoss = 0.
            model.train()
            for batch, [trainX, trainY] in enumerate(tqdm(train_dataloader, ncols=10)):
                train_n = len(trainX)
                train_N += train_n
                trainX = trainX.to(device)
                trainY = trainY.to(device).long()
                optimizer.zero_grad()
                predY = model(trainX)
                loss = SoftmaxWithXent(predY, trainY)

                loss.backward()  # get gradients on params
                optimizer.step()  # SGD update
                trainLoss += loss.detach().cpu().numpy()
            trainLoss /= train_N
            train_loss_list.append(trainLoss)
            test_N = 0.
            testLoss = 0.
            correct = 0.
            model.eval()
            for batch, [testX, testY] in enumerate(tqdm(test_dataloader, ncols=10)):
                test_n = len(testX)
                test_N += test_n
                testX = testX.to(device)
                testY = testY.to(device).long()
                predY = model(testX)
                loss = SoftmaxWithXent(predY, testY)
                testLoss += loss.detach().cpu().numpy()
                _, predicted = torch.max(predY.data, 1)
                correct += (predicted == testY).sum()
            testLoss /= test_N
            test_loss_list.append(testLoss)
            acc = correct / test_N
            acc_list.append(acc)
            print('epoch:{} train loss:{} testloss:{} acc:{}'.format(epoch, trainLoss, testLoss, acc))
        if not os.path.exists('./mnist_model'):
            os.mkdir('mnist_model')
        if not os.path.exists('./mnist_logs'):
            os.mkdir('mnist_logs')
        torch.save(model.state_dict(), './mnist_model/{}.pth'.format(resnet18_name[j]))
        print('模型已经保存')
        with open('./mnist_logs/{}.pkl'.format(resnet18_name[j]), 'wb') as file:
            pkl.dump([train_loss_list, test_loss_list, acc_list], file)
