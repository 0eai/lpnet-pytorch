from model import net, base
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import os
from time import time
from torch.optim import lr_scheduler

class lpnet():
    def __init__(self, numPoints, numClasses, model_name= None, net_file= None, base_file = None):
        self.numPoints = numPoints
        self.numClasses = numClasses
        self.model_name = model_name
        if self.model_name is None:
            self.model_name = 'lpnet'
        self.net_file = net_file
        self.base_file = base_file
        self.use_gpu= torch.cuda.is_available()
        self.load_model()
        if self.use_gpu:
            self.model = self.model.cuda()

    def train(self, dataloader, batchSize, epoch_start= 0, epochs= 25):
        self.dataloader = dataloader
        self.batchSize = batchSize
        self.epoch_start = epoch_start
        self.epochs = epochs
        if self.model_name == 'lpnet':
            self.criterion = nn.CrossEntropyLoss()
            # self.optimizer = optim.RMSprop(self.model.parameters(), lr=0.01, momentum=0.9)
            self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
            self.lrScheduler = lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1)
            self.train_net()
        else:
            self.criterion = nn.MSELoss()
            self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
            self.lrScheduler = lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1)
            self.train_base()

    def train_base(self):
        # since = time.time()
        for epoch in range(self.epoch_start, self.epochs):
            lossAver = []
            self.model.train(True)
            self.lrScheduler.step()
            start = time()

            for i, (XI, YI) in enumerate(self.dataloader):
                # print('%s/%s %s' % (i, times, time()-start))
                YI = np.array([el.numpy() for el in YI]).T
                if self.use_gpu:
                    x = Variable(XI.cuda(0))
                    y = Variable(torch.FloatTensor(YI).cuda(0), requires_grad=False)
                else:
                    x = Variable(XI)
                    y = Variable(torch.FloatTensor(YI), requires_grad=False)
                # Forward pass: Compute predicted y by passing x to the model
                y_pred = self.model(x)

                # Compute and print loss
                loss = 0.0
                if len(y_pred) == self.batchSize:
                    loss += 0.8 * nn.L1Loss().cuda()(y_pred[:][:2], y[:][:2])
                    loss += 0.2 * nn.L1Loss().cuda()(y_pred[:][2:], y[:][2:])
                    lossAver.append(loss.data[0])

                    # Zero gradients, perform a backward pass, and update the weights.
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    torch.save(self.model.state_dict(), self.model_name + '.pth')
                if i % 50 == 1:
                    with open(self.model_name + '.txt', 'a') as outF:
                        outF.write('train %s images, use %s seconds, loss %s\n' % (i*self.batchSize, time() - start, sum(lossAver[-50:]) / len(lossAver[-50:])))
            print ('%s %s %s\n' % (epoch, sum(lossAver) / len(lossAver), time()-start))
            with open(self.model_name + '.txt', 'a') as outF:
                outF.write('Epoch: %s %s %s\n' % (epoch, sum(lossAver) / len(lossAver), time()-start))
            torch.save(self.model.state_dict(), self.model_name + '.pth' + str(epoch))

    def train_net(self, model, criterion, optimizer, num_epochs=25):
        # since = time.time()
        for epoch in range(self.epoch_start, self.epochs):
            lossAver = []
            self.model.train(True)
            self.lrScheduler.step()
            start = time()

            for i, (XI, Y, labels, ims) in enumerate(self.dataloader):
                if not len(XI) == self.batchSize:
                    continue

                YI = [[int(ee) for ee in el.split('_')[:7]] for el in labels]
                Y = np.array([el.numpy() for el in Y]).T
                if self.use_gpu:
                    x = Variable(XI.cuda(0))
                    y = Variable(torch.FloatTensor(Y).cuda(0), requires_grad=False)
                else:
                    x = Variable(XI)
                    y = Variable(torch.FloatTensor(Y), requires_grad=False)
                # Forward pass: Compute predicted y by passing x to the model

                try:
                    fps_pred, y_pred = model(x)
                except:
                    continue

                # Compute and print loss
                loss = 0.0
                loss += 0.8 * nn.L1Loss().cuda()(fps_pred[:][:2], y[:][:2])
                loss += 0.2 * nn.L1Loss().cuda()(fps_pred[:][2:], y[:][2:])
                for j in range(7):
                    l = Variable(torch.LongTensor([el[j] for el in YI]).cuda(0))
                    loss += self.criterion(y_pred[j], l)

                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                try:
                    lossAver.append(loss.data[0])
                except:
                    pass

                if i % 50 == 1:
                    with open(self.model_name + '.txt', 'a') as outF:
                        outF.write('train %s images, use %s seconds, loss %s\n' % (i*self.batchSize, time() - start, sum(lossAver) / len(lossAver) if len(lossAver)>0 else 'NoLoss'))
                    torch.save(model.state_dict(), self.model_name + '.pth')
            print ('%s %s %s\n' % (epoch, sum(lossAver) / len(lossAver), time()-start))
            model.eval()
            count, correct, error, precision, avgTime = eval(model, testDirs)
            with open(self.model_name + '.txt', 'a') as outF:
                outF.write('%s %s %s\n' % (epoch, sum(lossAver) / len(lossAver), time() - start))
                outF.write('*** total %s error %s precision %s avgTime %s\n' % (count, error, precision, avgTime))
            torch.save(model.state_dict(), self.model_name + '.pth' + str(epoch))

    def load_model(self):
        if self.model_name == 'lpnet':
            if not self.net_file is None:
                if not os.path.isfile(self.net_file):
                    print ("fail to load existed model! Existing ...")
                    exit(0)
                print ("Load existed model! %s" % self.net_file)
                self.model = net.net(self.numPoints, self.numClasses)
                self.model = torch.nn.DataParallel(self.model, device_ids=range(torch.cuda.device_count()))
                self.model.load_state_dict(torch.load(self.net_file))
            else:
                self.model = net.net(self.numPoints, self.numClasses, self.base_file)
                if self.use_gpu:
                    self.model = torch.nn.DataParallel(self.model, device_ids=range(torch.cuda.device_count()))
        elif self.model_name == 'base':
            if not self.base_file is None:
                if not os.path.isfile(self.base_file):
                    print ("fail to load existed model! Existing ...")
                    exit(0)
                print ("Load existed model! %s" % self.base_file)
                self.model = base.base(self.numPoints)
                self.model = torch.nn.DataParallel(self.model, device_ids=range(torch.cuda.device_count()))
                self.model.load_state_dict(torch.load(self.base_file))
            else:
                self.model = base.base(self.numPoints)
                if self.use_gpu:
                    self.model = torch.nn.DataParallel(self.model, device_ids=range(torch.cuda.device_count()))
        else:
            print ("wrong model name! Existing ...")
            exit(0)
