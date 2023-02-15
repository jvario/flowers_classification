# PROGRAMMER:  Giannis Variozidis
# DATE CREATED: 13/02/2023                      
# REVISED DATE: 
# PURPOSE: Train VGG16 model for flower classification images.

# Imports here

import argparse
import torch
import torchvision
from torchvision import datasets, models, transforms
from torch import nn,optim
from collections import OrderedDict
import time
from time import sleep
import matplotlib.pyplot as plt
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def get_train_input_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("data_dir",type=str,default=None)
    parser.add_argument("--save_dir",type=str,default=None)
    parser.add_argument("--arch",type=str,default='vgg16')
    parser.add_argument("--learning_rate",type=int,default=0.001)
    parser.add_argument("--hidden_units",type=int,default=4096)
    parser.add_argument("--epochs",type=int,default=6)                 
    parser.add_argument("--gpu",type=bool,default=True)  
    
    return parser.parse_args()

def data_transformations_loaders():
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    batch_size = 64
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    validation_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])


    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)
    validation_datasets = datasets.ImageFolder(valid_dir, transform=validation_transforms)

    train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_datasets, batch_size=batch_size, shuffle=True)
    
    return train_datasets,test_datasets,validation_datasets,train_loader,test_loader,validation_loader
    
def pretrained_arch(train_datasets, learning_rate, arch, hidden_units):
    
     model = getattr(models, arch)
     pretrained_model = model(pretrained=True)

     params_to_update = []

     for param in pretrained_model.parameters():
        param.requires_grad = False

     classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088,hidden_units)),('dp1', nn.Dropout(0.3)),('r1', nn.ReLU()),('fcl2', nn.Linear(hidden_units,4096)),('dp2', nn.Dropout(0.3)),('r2', nn.ReLU()),('fcl3', nn.Linear(hidden_units,1000)),('out', nn.LogSoftmax(dim=1)),]))
    
     pretrained_model.classifier = classifier
     pretrained_model.class_idx_mapping = train_datasets.class_to_idx
     print(train_datasets.class_to_idx)
     print(pretrained_model.classifier)

     criterion = nn.NLLLoss()
     optimizer = optim.Adam(pretrained_model.classifier.parameters(), lr=learning_rate)

     pretrained_model.to(device)
       
     return pretrained_model, criterion, optimizer
    
def validation(model, testloader, criterion, device):
    train_losses, test_losses = [], []
    test_loss = 0
    accuracy = 0
    model.to(device)
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
        

    return test_loss, accuracy


def train(model, trainloader, validloader, epochs, print_every, criterion, optimizer, device='cuda'):
    steps = 0
    train_losses, test_losses = [], []
    for epoch in range(epochs):
        running_loss = 0
        for (images, labels) in trainloader:
            steps += 1
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model.forward(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if steps % print_every == 0:

                model.eval()

                with torch.no_grad():
                    validation_loss, accuracy = validation(model, validloader, criterion, device)

                print("Epoch: {}/{}.. ".format(epoch+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(validation_loss/len(validloader)),
                      "Validation Accuracy: {:.3f}".format((accuracy/len(validloader))*100))
                

                
                model.train()
                
                train_losses.append(running_loss/len(trainloader))
                test_losses.append(validation_loss/len(validloader))
                
                running_loss = 0
                
    return train_losses, test_losses, model
  
def test_model(test_loader, model):    
    correct = 0
    total = 0
    model.to(device)
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total

    
    

def main():
    start_time = time.time()
    in_arg = get_train_input_args()
    
    if in_arg.gpu == True:
        print('Found gpu use')
        device = "cuda"
    else:
        device = "cpu"
    
    print(in_arg)
    train_datasets,test_datasets,validation_datasets,train_loader,test_loader,validation_loader = data_transformations_loaders()
    
    pretrained_model, criterion, optimizer = pretrained_arch(train_datasets, in_arg.learning_rate, in_arg.arch, in_arg.hidden_units)
    
    train_losses, test_losses, model = train(model=pretrained_model, 
        trainloader=train_loader, 
        validloader=validation_loader,
        epochs=in_arg.epochs, 
        print_every=20, 
        criterion=criterion,
        optimizer=optimizer,
        device=device)

    print(model)
#   plt.plot(train_losses, label='Training loss')
#   plt.plot(test_losses, label='Validation loss')
#   plt.legend(frameon=False)
    print('Accuracy on Test_Data : '+str(test_model(test_loader,model))+'%')
    
    model.class_idx_mapping = train_datasets.class_to_idx
    
    torch.save({
            'model': model,
            'epoch': in_arg.epochs,
            'classifier': model.classifier,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'criterion' : criterion.state_dict(),
            'class_idx_mapping': model.class_idx_mapping,
            'arch': in_arg.arch,
            'print_every': 20,
            'device': in_arg.gpu
            }, in_arg.save_dir+'/'+'chpt1.pth')
    
    end_time = time.time()
    tot_time = end_time - start_time
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)) )

if __name__ == "__main__":
    main()
    