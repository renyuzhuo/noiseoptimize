import os
import pickle as pkl

import torch.nn.init
import torch.optim as optim
from torch import nn
from tqdm import tqdm

import LoadTinyImageNet
import Resnet34

if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
else:
    pass

device = Resnet34.device

if __name__ == '__main__':

    """
    resnet18_model = [Resnet34.getResnetPlus34A, Resnet34.getResnetN34A]
    resnet18_name = ['Resnet34.getResnetPlus34A', 'Resnet34.getResnetN34A']
    """
    resnet18_model = [Resnet34.getResnetN34A]

    resnet18_name = ['Resnet34.getResnetN34A']

    batch_size = 128
    loss = 0.
    train_dataloader, test_dataloader = LoadTinyImageNet.get_data(batch_size)

    for j in range(len(resnet18_model)):
        print('new model training:{}'.format(resnet18_name[j]))
        model = resnet18_model[j]()
        model = model.to(device)
        trainLoss = 0.
        testLoss = 0.
        learning_rate = 1e-2
        start_epoch = 0
        test_loss_list = []
        train_loss_list = []
        acc_list = []
        epoches = 80
        SoftmaxWithXent = nn.CrossEntropyLoss()
        # define optimization algorithm
        optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=learning_rate, weight_decay=5e-04)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)
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
            scheduler.step()
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
        if not os.path.exists('./SGDmnist_model'):
            os.mkdir('SGDmnist_model')
        if not os.path.exists('./SGDmnist_logs'):
            os.mkdir('SGDmnist_logs')
        torch.save(model.state_dict(), './SGDmnist_model/{}.pth'.format(resnet18_name[j]))
        print('模型已经保存')
        with open('./SGDmnist_logs/{}.pkl'.format(resnet18_name[j]), 'wb') as file:
            pkl.dump([train_loss_list, test_loss_list, acc_list], file)
