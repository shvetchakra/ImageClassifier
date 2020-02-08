import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import json
import argparse
import os

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('--save_dir', type = str, dest = 'checkpoint_dir', default = './')
    parser.add_argument('--arch', type = str, dest = 'model_name', default = 'densenet121')
    parser.add_argument('--learning_rate', type = float, default = .001)
    parser.add_argument('--hidden_units', type = int, default = 512)
    parser.add_argument('--epochs', type = int, default = 30)
    parser.add_argument('--gpu', action='store_true')

    return parser.parse_args()
args = argument_parser()
data_dir =  args.data_dir
checkpoint_dir = args.checkpoint_dir
model_name = args.model_name
learning_rate = args.learning_rate
hidden_units = args.hidden_units
epochs = args.epochs
gpu = args.gpu
device = 'cpu'
if gpu :
    device = 'cuda'

if os.path.exists(checkpoint_dir):
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pth')
#define the dataset variable and splitting it 
data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

#define the transformation
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
valid_transforms = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])


train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)


trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle=False)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)




def save_checkpoint(model,model_name,epoch,optimizer,criterion,valid_acc,checkpoint="checkpoint.pth"):
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {'input_size': 1024,
                  'output_size': 102,
                  'hidden_units': hidden_units,
                  'optimizer_state': optimizer.state_dict,
                  'criterion': criterion,
                  'epoch': epoch,
                  'model_name':model_name,
                  'val_acc': valid_acc,
                  'classifier':"Sequential",
                  'state_dict': model.state_dict()}
    if checkpoint_dir.is_dir():
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint)
        torch.save(checkpoint, checkpoint_path)
    else:
        torch.save(checkpoint, checkpoint)
def load_checkpoints(filepath):
    checkpoint = torch.load(filepath,map_location='cuda:0')
    model = get_model(checkpoint['model_name'])
    model.classifier = nn.Sequential(nn.Linear(checkpoint['input_size'],checkpoint['hidden_units'] ),
                                         nn.ReLU(),
                                         nn.Dropout(0.2),
                                         nn.Linear(checkpoint['hidden_units'], checkpoint['output_size']),
                                         nn.LogSoftmax(dim=1))
    model.optimizer = checkpoint['optimizer_state']
    model.criterion = checkpoint['criterion']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def get_model(model_name):
    if model_name.islower() == "vgg13":
        return models.vgg13(pretrained=True)
    if model_name.islower() == "alexnet":
        return models.alexnet(pretrained=True)
    if model_name.islower() == "squeezenet1_0":
        return models.squeezenet1_0(pretrained=True)
    if model_name.islower() == "resnet18":
        return models.resnet18(pretrained=True)
    return models.densenet121(pretrained=True)

def train_model(load_checkpoint="checkpoint.pth",save_checkpoint="checkpoint.pth"):
    

            
    model = get_model(model_name)
    for param in model.parameters():
        param.requires_grad = False
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pth')
    
    if os.path.exists(checkpoint_path):
        model = load_checkpoints(checkpoint_path)
        criterion = model.criterion
        model.cuda()
        '''optimizer = optim.Adam(model.parameters(),)
        optimizer.load_state_dict(model.optimizer)
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()'''
        
        
    else:
        
        model.classifier = nn.Sequential(nn.Linear(1024, 512),
                                             nn.ReLU(),
                                             nn.Dropout(0.2),
                                             nn.Linear(512, 102),
                                             nn.LogSoftmax(dim=1))
        criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    #model = models.densenet121(pretrained=True)
    #state_dict = torch.load('checkpoint.pth')   
    #model.load_state_dict(state_dict)


    
    steps = 0
    running_loss = 0
    print_every = 10
    prev_acc = 0
    accuracy = 0
    model.to(device)
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        valid_acc = accuracy/len(validloader)

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Valid loss: {valid_loss/len(validloader):.3f}.. "
                      f"Validation accuracy: {valid_acc:.3f}")
                running_loss = 0
                model.train()
        if valid_acc-prev_acc < .1 and valid_acc > .9:
            break
        else:
            prev_acc = valid_acc
    save_checkpoint(model,model_name,epoch,optimizer,criterion,valid_acc,"checkpoint.pth")


train_model()