# Imports here
import argparse
import torch
import torchvision
from torchvision import datasets, models, transforms
import time
from time import sleep
import matplotlib.pyplot as plt
import numpy as np
import json
from PIL import Image


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


def get_pred_input_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("input",type=str,default=None)
    parser.add_argument("checkpoint",type=str,default=None)
    parser.add_argument("--top_k",type=int,default='5')
    parser.add_argument("--category_names",type=str,default='cat_to_name.json')
    parser.add_argument("--gpu",type=bool,default=True)  
    
    return parser.parse_args()

def load_checkpoint(model_chpt):
    chpt  = torch.load(model_chpt)
    model = chpt['model']
    model.load_state_dict(chpt['state_dict'])
    
    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img = Image.open(image)
    width,height = img.size
    print(width,height)
    n_size = 256,999999999
    img.thumbnail(n_size)
    width,height = img.size
    
    print(width,height)
    
    left = (width - 224) / 2
    right = (width + 224) / 2
    top = (height - 224) / 2
    bottom = (height + 224) / 2
    img = img.crop((left, top, right, bottom))
    
    
    
    nump_img = np.array(img)/255
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    nump_img = (nump_img - mean) / std
    
    nump_img = nump_img.transpose(2, 0, 1)
    
    
    return nump_img

def imshow(image, ax, title):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    ax.set_title(title)
    ax.imshow(image)
    
    return ax



def predict(image_path, model,train_datasets,device,topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    class_idx_mapping = train_datasets.class_to_idx
    idx_class_mapping = {v: k for k, v in class_idx_mapping.items()}
    
    
    if torch.cuda.is_available():
        model.to('cuda')
        
    np_image = process_image(image_path)
    np_image = np.resize(np_image,(1, 3, 224, 224))

    image_tensor = torch.from_numpy(np_image).type(torch.FloatTensor).to(device)
    
    model.eval()  
    
    with torch.no_grad():
        outputs = model(image_tensor)
    
    probs, indices = torch.topk(outputs, topk)
    probs = probs.exp().cpu().numpy()[0]
    indices = indices.cpu().numpy()[0]
    classes = [idx_class_mapping[index] for index in indices]
    
    return probs,classes


def main():
    start_time = time.time()
    
    in_arg = get_pred_input_args()    
    
    
    with open(in_arg.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    train_datasets,test_datasets,validation_datasets,train_loader,test_loader,validation_loader = data_transformations_loaders()
    loaded_model = load_checkpoint(model_chpt = in_arg.checkpoint)
    
    if in_arg.gpu == True:
        print('Found gpu use')
        device = "cuda"
    else:
        device = "cpu"
    
    probs,classes = predict(in_arg.input,loaded_model,train_datasets,device,in_arg.top_k)
    
   

    class_names = [cat_to_name[c] for c in classes]
    
    top_prob = max(probs)
    top_class = classes[0]
    top_class = cat_to_name[str(top_class)]
    
    
    end_time = time.time()
    tot_time = end_time - start_time
    print(f'\n Probalities: {probs}, \n Classes: {class_names}, \n Top-Probability: {top_prob}, \n Top-Class_title: {top_class}') 
    
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)) )

if __name__ == "__main__":
    main()
    
