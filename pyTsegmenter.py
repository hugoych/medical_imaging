# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 09:46:14 2018

@author: hugol
"""

import torch
from torch.autograd import Variable
import torch.optim as optim

import time
from datasets import SegDataset,AdvDataset,resize, TargetDataset
from nets import SegmenterCNN,DiscriminatorCNN


#%%



#%% TESTS
        
from skimage.io import imshow

Seg = SegDataset('Train/Seg_train',transform=resize)
Tar = TargetDataset('Target',transform=resize)
b = Tar.__getitem__(150)
a = Seg.__getitem__(166)
#print(os.listdir('Train/Seg_train/X_S')[1])

#b = imread('Train/Seg_train/X_S/IM_000032.jpg')
imshow(a['segment'][0])
        
#%%




    
def get_train_loader(dataset,batch_size):
    train_seg_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size = batch_size,
                                          shuffle = True,
                                          num_workers=0)
    return train_seg_loader


class Segmenter(object):
    def __init__(self, epoch=20,lr=1e-5,batch_size = 1, dataset=SegDataset('Train/Seg_train',transform=resize), gpu_mode=True):
        # parameters
        self.epoch = epoch
        self.learning_rate = lr
        self.batch_size = batch_size
        self.dataset = dataset
        self.gpu_mode = gpu_mode
        self.input_dim = 150
        
        # load dataset
        # networks init
        self.net = SegmenterCNN()
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)
        
        def DSC(logits,labels):
            pred = logits.view(2,self.batch_size,self.input_dim-18,self.input_dim-18)
            labels =labels.view(2,self.batch_size,self.input_dim-18,self.input_dim-18)
            precision = torch.sum(pred[1]*labels[1])/torch.sum(pred[1])
            recall = torch.sum(pred[1]*labels[1])/torch.sum(labels[1])

            
            return -2*(precision*recall)/(precision+recall)
    
        self.loss = DSC
        
        
        
    
        
    def train(self):
        net = self.net.cuda()
        batch_size=self.batch_size
        n_epochs=self.epoch
        #Print all of the hyperparameters of the training iteration:
        print("===== HYPERPARAMETERS =====")
        print("batch_size=", batch_size)
        print("epochs=", n_epochs)
        print("learning_rate=", self.learning_rate)
        print("=" * 30)
        val_loader = torch.utils.data.DataLoader(SegDataset('Val/Seg_val',transform = resize),
                                         batch_size = batch_size,
                                         shuffle = True,
                                         num_workers = 0)
        #Get training data
        train_loader = get_train_loader(self.dataset, batch_size)
        n_batches = len(train_loader)
        
        #Create our loss and optimizer functions
        loss, optimizer = self.loss, self.optimizer
        
        #Time for printing
        training_start_time = time.time()
        
        #Loop for n_epochs
        for epoch in range(n_epochs):
            running_loss = 0.0
            print_every = n_batches // 10
            start_time = time.time()
            total_train_loss = 0
            
            for i, data in enumerate(train_loader, 0):
                
                #Get inputs
                #Wrap them in a Variable object
                if self.gpu_mode:
                    inputs, labels = torch.tensor(data['image'], dtype=torch.float).cuda() ,torch.tensor(data['segment'], dtype=torch.float).cuda()
                    inputs, labels = Variable(inputs), Variable(labels)
                else:
                    inputs, labels = torch.tensor(data['image'], dtype=torch.float) ,torch.tensor(data['segment'], dtype=torch.float)
                    inputs, labels = Variable(inputs), Variable(labels)
                
                #Set the parameter gradients to zero
                optimizer.zero_grad()

                
                #Forward pass, backward pass, optimize
                outputs = net(inputs)
                loss_size = loss(outputs, labels)
                #loss_size.backward()
                #optimizer.step()
                loss_size.backward()
                
                #print(list(net.parameters())[0].grad.mean())
                optimizer.step()
                #Print statistics
                running_loss += loss_size.item()
                total_train_loss += loss_size.item()
                
                #Print every 10th batch of an epoch
                if (i + 1) % (print_every + 1) == 0:
                    print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                            epoch+1, int(100 * (i+1) / n_batches), running_loss / print_every, time.time() - start_time))
                    print('data max', outputs[0,0].max(),outputs[0,1].max())
                    
                    #Reset running loss and time
                    running_loss = 0.0
                    start_time = time.time()
                
            #At the end of the epoch, do a pass on the validation set
            #test()
            total_val_loss = 0
            
            for data in val_loader:
                
                inputs, labels = torch.tensor(data['image'], dtype=torch.float).cuda() ,torch.tensor(data['segment'], dtype=torch.float).cuda()
                #Wrap them in a Variable object
                inputs, labels = Variable(inputs), Variable(labels)
                
                #Forward pass
                val_outputs = net(inputs)
                val_loss_size = loss(val_outputs, labels)
                total_val_loss += val_loss_size.data[0]
                
            print("Validation loss = {:.2f}".format(total_val_loss / len(val_loader)))
            
            print("Training finished, took {:.2f}s".format(time.time() - training_start_time))
    def test(self):
        pred = self.net.cuda()(torch.tensor(a['image'].reshape(1,3,self.input_dim,self.input_dim),dtype=torch.float).cuda())
        imshow(pred.cpu().detach().numpy().reshape(2,self.input_dim-18,self.input_dim-18)[1])
    
    def save(self,PATH_seg):
        torch.save(self.net.state_dict(), PATH_seg)
        
    def load(self,PATH_seg):
        
        self.net.load_state_dict(torch.load(PATH_seg))
        self.net.eval()




class UDA(object):
    def __init__(self, epoch=45,lr_seg=0.0001,lr_adv=0.001,batch_size = 1, dataset=AdvDataset('Train/Adv_train',transform=resize), gpu_mode=True):
        # parameters
        self.epoch = epoch
        self.lr_seg = lr_seg
        self.lr_adv = lr_adv
        self.batch_size = batch_size
        self.dataset = dataset
        self.gpu_mode = gpu_mode
        self.input_dim = 150
        
        # load dataset
        # networks init
        self.seg = SegmenterCNN(adv=True)
        self.adv = DiscriminatorCNN()
        self.optimizer_seg = optim.Adam(self.seg.parameters(), lr=self.lr_seg)
        self.optimizer_adv = optim.Adam(self.adv.parameters(), lr=self.lr_adv)
        
    def DSC(logits,labels):
            eps=10e-5
            pred = logits.view(2,self.batch_size,self.input_dim-18,self.input_dim-18)
            labels =labels.view(2,self.batch_size,self.input_dim-18,self.input_dim-18)
            precision = torch.sum(pred[1]*labels[1])/torch.sum(pred[1])
            
            recall = torch.sum(pred[1]*labels[1])/torch.sum(labels[1])
            
            return -2*(precision*recall)/(precision+recall+eps)
    
    
        
        
        
        
    
        
    def train(self):
        seg = self.seg.cuda()
        adv = self.adv.cuda()
        batch_size=self.batch_size
        n_epochs=self.epoch
        #Print all of the hyperparameters of the training iteration:
        print("===== HYPERPARAMETERS =====")
        print("batch_size=", batch_size)
        print("epochs=", n_epochs)
        print("learning_rate_seg=", self.lr_seg)
        print("learning_rate_adv=", self.lr_adv)
        print("=" * 30)
        
        "LOADER A MODIFER"     
        val_loader = torch.utils.data.DataLoader(AdvDataset('Val/Adv_val',transform = resize),
                                         batch_size = batch_size,
                                         shuffle = True,
                                         num_workers = 0)
        #Get training data
        train_loader = get_train_loader(self.dataset, batch_size)
        
        n_batches = len(train_loader)
        


        #Create our loss and optimizer functions
        loss_dis , loss_seg = torch.nn.BCELoss(), DSC
        optimizer_seg, optimizer_adv = self.optimizer_seg, self.optimizer_adv
        
        def loss_seg_adv(loss_adv,logits,gt,alpha):
            return loss_seg(logits,gt) -alpha*loss_adv
        
        #Time for printing
        training_start_time = time.time()
        
        #Loop for n_epochs
        alpha = 0
        e1=15
        e2=40
        alpha_max = 0.1
        
        for epoch in range(n_epochs):
            running_loss = 0.0
            running_loss_dis = 0.0
            print_every = n_batches // 10
            start_time = time.time()
            total_train_loss = 0
            total_train_loss_dis = 0
            G=0
            F=0
            if epoch >= e1 and epoch <= e2:
                alpha = alpha_max*(epoch-e1)/(e2-1)
            
            for i, data in enumerate(train_loader, 0):
               
                
                
                #Get inputs
                #Wrap them in a Variable object
                if self.gpu_mode:
                    input_seg, label_seg = torch.as_tensor(data[1]['image'], dtype=torch.float).cuda() ,torch.as_tensor(data[1]['segment'], dtype=torch.float).cuda()
                    input_adv = torch.as_tensor(data[0]['image'], dtype=torch.float).cuda()
                    label_adv = torch.as_tensor(data[0]['source'], dtype=torch.float).cuda()
                    input_seg, input_adv, label_seg, label_adv = Variable(input_seg), Variable(input_adv),Variable(label_seg), Variable(label_adv)
                else:
                    input_seg, label_seg = torch.as_tensor(data[1]['image'], dtype=torch.float) ,torch.as_tensor(data[1]['segment'], dtype=torch.float)
                    input_adv = torch.as_tensor(data[0]['image'], dtype=torch.float)
                    label_adv = torch.as_tensor(data[0]['source'], dtype=torch.float)
                    input_seg, input_adv, label_seg, label_adv = Variable(input_seg), Variable(input_adv),Variable(label_seg), Variable(label_adv)
                
                #Set the parameter gradients to zero
                
                for k in range(2):
                    #optimizer_seg.zero_grad()
                    #optimizer_adv.zero_grad()
                    if k == 0 :
                        optimizer_adv.zero_grad()
                        
                        output_dis = adv(seg(input_adv)[1])
                        loss_discriminator = loss_dis(output_dis[0],label_adv)
                        
                        loss_discriminator.backward()
                        optimizer_adv.step()
                        running_loss_dis += loss_discriminator.item()
                        total_train_loss_dis += loss_discriminator.item()
                        loss_adv_value =loss_discriminator.item()
                        if output_dis[0].argmax() == label_adv.argmax():
                            G+=1
                        else:
                            F+=1
                        
                    if k == 1 :
                        optimizer_seg.zero_grad()
                        output_seg = seg(input_seg)[0]
                        loss_adversarial = loss_seg_adv(loss_adv_value,output_seg,label_seg,alpha)
                        loss_adversarial.backward()
                        optimizer_seg.step()
                        running_loss += loss_adversarial.item()
                        total_train_loss += loss_adversarial.item()
                    
                
                #Forward pass, backward pass, optimize
                                   
                
                #print(list(net.parameters())[0].grad.mean())
                
                #Print statistics

                
                #Print every 10th batch of an epoch
                if (i + 1) % (print_every + 1) == 0:
                    print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                            epoch+1, int(100 * (i+1) / n_batches), running_loss / print_every, time.time() - start_time))
                    print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                            epoch+1, int(100 * (i+1) / n_batches), running_loss_dis / print_every, time.time() - start_time))
                    print('data max', output_seg[0,0].mean(),output_seg[0,1].mean())


                    
                    #Reset running loss and time
                    running_loss = 0.0
                    running_loss_dis = 0.0
                    start_time = time.time()
             
            print('Number of well prediction', G/(G+F))
            #At the end of the epoch, do a pass on the validation set
            #test()
            total_val_loss = 0
            total_val_loss_dis = 0
            
            for data in val_loader:
                
                if self.gpu_mode:
                    input_seg, label_seg = torch.as_tensor(data[1]['image'], dtype=torch.float).cuda() ,torch.as_tensor(data[1]['segment'], dtype=torch.float).cuda()
                    input_adv = torch.as_tensor(data[0]['image'], dtype=torch.float).cuda()
                    label_adv = torch.as_tensor(data[0]['source'], dtype=torch.float).cuda()
                    input_seg, input_adv, label_seg, label_adv = Variable(input_seg), Variable(input_adv),Variable(label_seg), Variable(label_adv)
                else:
                    input_seg, label_seg = torch.tensor(data[1]['image'], dtype=torch.float) ,torch.tensor(data[1]['segment'], dtype=torch.float)
                    input_adv = torch.tensor(data[0]['image'], dtype=torch.float)
                    label_adv = torch.tensor(data[0]['source'], dtype=torch.float)
                    input_seg, input_adv, label_seg, label_adv = Variable(input_seg), Variable(input_adv),Variable(label_seg), Variable(label_adv)
                
                #Forward pass
                
                val_output_seg = seg(input_seg)[0]
                #output_dis = adv(seg(input_adv)[1])
                
                
                #loss_discriminator = loss_dis(output_dis,label_adv)
                loss_segmenter = loss_seg(val_output_seg,label_seg)

                
                total_val_loss += loss_segmenter.item()
                #total_val_loss_dis += loss_discriminator.item()
                
            print("Validation loss segmenter = {:.2f}".format(total_val_loss / len(val_loader)))
            print("Validation loss discriminator= {:.2f}".format(total_val_loss_dis / len(val_loader)))

            
            print("Training finished, took {:.2f}s".format(time.time() - training_start_time))
                    
                
    def test(self,):
        pred = self.seg.cuda()(torch.tensor(a['image'].reshape(1,3,self.input_dim,self.input_dim),dtype=torch.float).cuda())[0]
        imshow(pred.cpu().detach().numpy().reshape(2,self.input_dim-18,self.input_dim-18)[1])
        pred2 = self.seg.cuda()(torch.tensor(b['image'].reshape(1,3,self.input_dim,self.input_dim),dtype=torch.float).cuda())[0]
        imshow(pred2.cpu().detach().numpy().reshape(2,self.input_dim-18,self.input_dim-18)[1])
    
    def save(self,PATH_seg,PATH_adv):
        torch.save(self.seg.state_dict(), PATH_seg)
        torch.save(self.adv.state_dict(),PATH_adv)
    
    def load(self,PATH_seg,PATH_adv):
                
        self.seg.load_state_dict(torch.load(PATH_seg))
        self.seg.eval()
        
        self.adv.load_state_dict(torch.load(PATH_adv))
        self.adv.eval()
        
    def get_DSC_on_T(self):
        total_loss = 0
        dataloader = torch.utils.data.DataLoader(Tar,
                                             batch_size = 1,
                                             shuffle = True,
                                             num_workers = 0)
        for data in dataloader:
            input_seg, label_seg = torch.as_tensor(data['image'], dtype=torch.float).cuda() ,torch.as_tensor(data['segment'], dtype=torch.float).cuda()
            output = self.seg(input_seg)[0]
            loss = DSC(output,label_seg)
            total_loss += loss.item()
            
        return(total_loss/len(dataloader))


input_dim =150

def DSC(logits,labels):
            eps=10e-5
            pred = logits.view(2,1,input_dim-18,input_dim-18)
            labels =labels.view(2,1,input_dim-18,input_dim-18)
            precision = torch.sum(pred[1]*labels[1])/torch.sum(pred[1])
            
            recall = torch.sum(pred[1]*labels[1])/torch.sum(labels[1])
            
            return -2*(precision*recall)/(precision+recall+eps)
    

def get_DSC_on_T(seg):
    total_loss = 0
    dataloader = torch.utils.data.DataLoader(Tar,
                                         batch_size = 1,
                                         shuffle = True,
                                         num_workers = 0)
    for data in dataloader:
        input_seg, label_seg = torch.as_tensor(data['image'], dtype=torch.float).cuda() ,torch.as_tensor(data['segment'], dtype=torch.float).cuda()
        output = seg.net(input_seg)
        loss = DSC(output,label_seg)
        total_loss += loss.item()
        
    return(total_loss/len(dataloader))
        
