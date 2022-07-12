#import
from torchvision import datasets, transforms,models
import numpy as np
import torch
from PIL import Image
from torch import optim,nn
import os
import shutil
import pandas as pd

def load_data(data_dir):
	#Define directories for trainning, validation and testing
	train_dir=data_dir + '/train'
	valid_dir=data_dir + '/valid'
	test_dir=data_dir + '/test'
	#Define transforms for sets
	train_transforms = transforms.Compose([transforms.Resize(256),
									   transforms.RandomRotation(30),
									   transforms.RandomResizedCrop(224),
									   transforms.RandomHorizontalFlip(),
									   transforms.ToTensor(),
									   transforms.Normalize([0.485, 0.456, 0.406], 
															[0.229, 0.224, 0.225])])

	test_transforms = transforms.Compose([transforms.Resize(256),
									   transforms.CenterCrop(224),
									   transforms.ToTensor(),
									   transforms.Normalize([0.485, 0.456, 0.406], 
															[0.229, 0.224, 0.225])])

	valid_transforms = transforms.Compose([transforms.Resize(256),
									   transforms.CenterCrop(224),
									   transforms.ToTensor(),
									   transforms.Normalize([0.485, 0.456, 0.406], 
															[0.229, 0.224, 0.225])])
	#Load the datasets with images
	train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
	test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)
	valid_data = datasets.ImageFolder(data_dir + '/test', transform=valid_transforms)
	dataloader={'train':torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True),
				'test':torch.utils.data.DataLoader(test_data, batch_size=32,shuffle=True),
				'valid':torch.utils.data.DataLoader(valid_data, batch_size=32, shuffle=True)}

	return dataloader,train_data.class_to_idx

def saving_ck(save_path,epochs,loss,Model_dict,Model_classifier,optimizer_state,class_idx,criterion,input_layer,hidden_layer,output_layer,dropout,model_name):
	checkpoint={
		'epoch':epochs,
		'model_state_dict':Model_dict,
		'optimizer_state_dict':optimizer_state,
		'classifier':Model_classifier,
		'loss':loss,
		'class_to_idx':class_idx,
		'criterion':criterion,
		'input_layer':input_layer,
		'hidden_layer':hidden_layer,
		'output_layer':output_layer,
		'model_name':model_name,
		'dropout':dropout,
	}
	torch.save(checkpoint,save_path)
	print('Checkpoint saved')

def loading_ck(Fpathname,criterion,device):
	print(Fpathname)
	if os.path.isfile(Fpathname):
		checkpoint=torch.load(Fpathname,map_location=lambda storage, loc: storage)
		Model,input_lay,nom_mod=choice_model(checkpoint['model_name'])
		#Model.classifier=torch.nn.Sequential(*build_sequence(checkpoint['input_layer'],checkpoint['output_layer'],checkpoint['hidden_layer'],checkpoint['dropout']))
		current_epoch=checkpoint['epoch']
		Model.class_to_idx=checkpoint['class_to_idx']
		for parametres in Model.parameters():
			parametres.requires_grad=False
		Model.classifier=checkpoint['classifier']
		Model.load_state_dict(checkpoint['model_state_dict'])
		Model.to(device)
		Optimizer = torch.optim.Adam(Model.classifier.parameters()) 
		Optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		Loss=checkpoint['loss']
		criterion=checkpoint['criterion']
		return current_epoch,Model,Optimizer,Loss,criterion,checkpoint['input_layer'],checkpoint['model_name']
		print('Checkpoint loaded')
	else:
		print('No such checkpoint')

def process_image(image):
	width,height=image.size
	image = image.resize((256, int(256*(height/width))) if width < height else (int(256*(width/height)), 256))
	width,height=image.size
	crop_size= ((width - 224)/2,(height - 224)/2,(width + 224)/2,(height + 224)/2)
	image = image.crop(crop_size)
	image = np.array(image)
	image=image/255
	Means_TO=np.array([0.485, 0.456, 0.406])
	Std_TO=np.array([0.229, 0.224, 0.225])
	image=(image-Means_TO)/Std_TO
	image=image.transpose(2,0,1)
	image=torch.from_numpy(image)
	image=image.float()
	return image


def build_sequence(input_layer,output_layer,hidden_layer,dropout):
    if len(hidden_layer)>0:
        sequ=[nn.Linear(input_layer,hidden_layer[0]),nn.ReLU(),nn.Dropout(dropout[0])]
        for k in range(len(hidden_layer)-1):
            sequ=sequ
            [nn.Linear(hidden_layer[k],hidden_layer[k+1]),nn.ReLU(),nn.Dropout(dropout[k+1])]

        sequ=sequ+[nn.Linear(hidden_layer[len(hidden_layer)-1],output_layer),nn.LogSoftmax(dim=1)]

    else:
        sequ=[nn.Linear(input_layer,output_layer),nn.LogSoftmax(dim=1)]

    return sequ
#Model chosed by the user
def choice_model(modl):
    #Three models
    if modl=='alexnet':
        return models.alexnet(pretrained=True), 9216,'alexnet'
    elif modl=='densenet169':
        return models.densenet169(pretrained=True),1664,'densenet169'
    else:
        return models.vgg16(pretrained=True),25088,'vgg16'

#Testing the model
def test_model(Model,testloader,device,criterion):
    test_loss = 0
    test_accuracy = 0
    Model.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logresult = Model.forward(inputs)
            test_loss += criterion(logresult, labels).item()
            result = torch.exp(logresult)
            testeur = torch.max(result, dim=1)[1]== labels
            test_accuracy += torch.mean(testeur.type(torch.FloatTensor)).item()
        print(f"test loss: {test_loss/len(testloader):.3f}.. "
        f"test accuracy: {test_accuracy/len(testloader):.3f}")
    return test_accuracy/len(testloader)
def train_model(epochs,dataloaders,device,criterion,optimizer,Model,current_epoch):
    for epoch in range(epochs):
        training_loss=0
        for inputs, labels in dataloaders['train']:
            optimizer.zero_grad()   
            inputs, labels = inputs.to(device), labels.to(device)        
            logresult = Model.forward(inputs)
            loss = criterion(logresult, labels)
            loss.backward()
            optimizer.step()     
            training_loss += loss.item()

        valid_loss = 0
        accuracy = 0
        Model.eval()
        with torch.no_grad():
            for inputs, labels in dataloaders['valid']:
                inputs, labels = inputs.to(device), labels.to(device)
                logresult = Model.forward(inputs)
                valid_loss += criterion(logresult, labels).item()
                result = torch.exp(logresult)
                testeur = torch.max(result, dim=1)[1]== labels
                accuracy += torch.mean(testeur.type(torch.FloatTensor)).item()

        print(f"Epoch {current_epoch+epoch+1}.. "
                f"training loss: {training_loss/len(dataloaders['train']):.3f}.. "
                f"validation loss: {valid_loss/len(dataloaders['valid']):.3f}.. "
                f"valid accuracy: {accuracy/len(dataloaders['valid']):.3f}")    
        Model.train()
    return Model


#Trainning function
def trainning(save_path,modl,output_lay,hidden_layer,dropout,device,learn_rate,epochs,pathdir,chekpoint):
    dataloaders,class_to_idx=load_data(pathdir)
    criterion = nn.NLLLoss()  
    current_epoch=0
    name_mod=modl
    if chekpoint!='.pth':
        current_epoch,Model,optimizer,loss,criterion,input_lay,name_mod=loading_ck(chekpoint,criterion,device)
        Model.train()
    else:
        Model,input_lay,name_mod=choice_model(modl)   
        Model.classifier=nn.Sequential(*build_sequence(input_lay,output_lay,hidden_layer,dropout))   
        Model.to(device)
        optimizer = optim.Adam(Model.classifier.parameters(), lr=learn_rate)
    print("************Training session**************\n")
    Model=train_model(epochs,dataloaders,device,criterion,optimizer,Model,current_epoch)   
    #print("\n ************Testing session************** \n")
    #acc_test=test_model(Model,dataloaders['test'],device,criterion)
    #saving_ck(save_path,epochs+current_epoch,loss,Model.state_dict(),Model.classifier,optimizer.state_dict(),class_to_idx,criterion,input_lay,hidden_layer,output_lay,dropout,name_mod)
    print("\n ************Model Summary************ \n")
    print("Name of the Model: {}".format(name_mod))
    #print("Epochs: {}".format(epoc))
    #print("Accuracy on test set: {}%".format(100*acc_test))
    Model.class_to_idx=class_to_idx
    return Model

#Launch trainning
save_path='/'
#model=['vgg16','densenet169','alexnet']
model='densenet169'
output_layer=120
hidden_layer=[1000]
dropout=[0.1]
device='cuda'
#learn_rate=[0.000001,0.000075,0.00005,0.000025,0.00001,0.00075,0.0005,0.00025,0.0001,0.0075,0.005,0.0025,0.001]
learn_rate=0.00025
epochs=10
data_directory='/kaggle/input/images/images/images'
Pcheckpoint=''
trainni=1
if trainni==1:
    Model=trainning(save_path+"checkpoint.pth",model,output_layer,hidden_layer,dropout,device,learn_rate,epochs,data_directory,Pcheckpoint + ".pth")
    trainni=trainni+1
def make_predictions():
    directory='/kaggle/input/dog-breed-identification/test/'
    device = "cuda"
    topk=120
    ch_path='/kaggle/densenet16930.pth'
    def predict(image_path, model, topk,device):
        model.eval()
        image_prepo=process_image(Image.open(image_path)).to(device)
        dictio={v: k for k, v in model.class_to_idx.items()}
        with torch.no_grad():
            logresult = model.forward(image_prepo.unsqueeze(0))
            result=torch.exp(logresult)
            legends=np.array(torch.topk(result,topk,dim=1)[1][0].cpu())
            proba=np.array(torch.topk(result,topk,dim=1)[0][0].cpu())
            legends=[dictio[x] for x in legends]
            proba=[x for x in proba]
            resultat={}
            for i in range(len(proba)):
                resultat[legends[i]]=proba[i]
        
        return resultat

    def Make_a_pred(image_path,ch_path,topk,device):
        criterion = nn.NLLLoss() 
        Model.to(device)
        return predict(image_path,Model,topk,device)
    
    submission=pd.read_csv('/kaggle/input/dog-breed-identification/sample_submission.csv')
    submission=submission.drop(submission.index, axis=0)
    
    for filename in os.listdir(directory):
        resultat=Make_a_pred(directory+filename,ch_path,topk,device)
        resultat['id']=filename.strip('.jpg')
        submission=submission.append(resultat, ignore_index=True)
    submission.to_csv('csv_to_submit.csv', index = False)
    return submission   

#Lauch submission file generation
submission=make_predictions() 
submission.to_csv('csv_to_submit.csv', index = False)
    