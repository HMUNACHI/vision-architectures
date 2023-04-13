import torch
import numpy as np
import torch.nn as nn
from torch import optim

class Trainer:
    def __init__(self, model, device):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(device)

    def train(self, epochs, name, train_loader, valid_loader):
        # training the model
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr = 0.01)
        valid_loss_min = np.Inf #set initial mean to infinity

        for epoch in range(epochs):
            # monitor training loss
            train_loss = 0.0
            valid_loss = 0.0
            
            ###################
            # train the model #
            ###################
            # prep model for training
            self.model.train() 
            for data, target in train_loader:
                # moving data and target to GPU if available
                data, target = data.to(self.device), target.to(self.device)
                # clear the gradients of all optimized variables
                optimizer.zero_grad()
                # forward pass: compute predicted outputs by passing inputs to the self.model
                output = self.model(data)
                # calculate the loss
                loss = criterion(output, target)
                # backward pass: compute gradient of the loss with respect to self.model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                optimizer.step()
                # update running training loss
                train_loss += loss.item()*data.size(0)
                
            ######################    
            # validate the self.model #
            ######################
            self.model.eval() # prep self.model for evaluation
            for data, target in valid_loader:
                # moving data and target to GPU if available
                data, target = data.to(self.device), target.to(self.device)
                # forward pass: compute predicted outputs by passing inputs to the self.model
                output = self.model(data)
                # calculate the loss
                loss = criterion(output, target)
                # update running validation loss 
                valid_loss += loss.item()*data.size(0)
                
            # print training/validation statistics 
            # calculate average loss over an epoch
            train_loss = train_loss/len(train_loader.sampler)
            valid_loss = valid_loss/len(valid_loader.sampler)
            
            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
                epoch+1, 
                train_loss,
                valid_loss
                ))
            
            # save self.model if validation loss has decreased (Early stopping)
            if valid_loss <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving self.model ...'.format(
                valid_loss_min,
                valid_loss))
                torch.save(self.model.state_dict(), name)
                valid_loss_min = valid_loss


    def test(self, test_loader, name, batch_size, classes):
        self.model.load_state_dict(torch.load(name))
        # initialize lists to monitor test loss and accuracy
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        criterion = nn.CrossEntropyLoss()
        test_loss = 0.0
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))

        self.model.eval() # prep self.model for *evaluation*

        for data, target in test_loader:
            # moving data and target to GPU if available
            data, target = data.to(device), target.to(device)
            # forward pass: compute predicted outputs by passing inputs to the self.model
            output = self.model(data)
            # calculate the loss
            loss = criterion(output, target)
            # update test loss 
            test_loss += loss.item()*data.size(0)
            # convert output probabilities to predicted class
            _, pred = torch.max(output, 1)
            # compare predictions to true label
            correct = np.squeeze(pred.eq(target.data.view_as(pred)))
            # calculate test accuracy for each object class
            for i in range(batch_size):
                label = target.data[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1

        # calculate and print avg test loss
        test_loss = test_loss/len(test_loader.dataset)
        print('Test Loss: {:.6f}\n'.format(test_loss))

        for i in range(10):
            if class_total[i] > 0:
                print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                    str(i), 100 * class_correct[i] / class_total[i],
                    np.sum(class_correct[i]), np.sum(class_total[i])))
            else:
                print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

        print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
            100. * np.sum(class_correct) / np.sum(class_total),
            np.sum(class_correct), np.sum(class_total)))