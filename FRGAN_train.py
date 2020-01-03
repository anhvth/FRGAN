import sys 
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from data_modules import arrange_data, get_poisoned_data, test_model
from FRGAN import Generator, Discriminator_F, Discriminator_R, weights_init_normal

import math
from argparse import Namespace


'''
def train_model() : training FR-GAN structure with given data.
The input arguments are three datasets (train/val/test tensors; Namespace type), a train option set (Namespace type), and two lambda values to control the impacts of discriminators.
The 'train_tensors' contains three data tensors: XS_train, y_train, and s1_train.
Similarly, 'val_tensors' and 'test_tensors' contain three tensors each.

After the FR-GAN training, the test accuracy and the value of disparate impact will be returned to the caller.
'''
def train_model(train_tensors, val_tensors, test_tensors, train_opt, lambda_f, lambda_r):
    
    XS_train = train_tensors.XS_train
    y_train = train_tensors.y_train
    s1_train = train_tensors.s1_train
    
    XS_val = val_tensors.XS_val
    y_val = val_tensors.y_val
    s1_val = val_tensors.s1_val
    
    XS_test = test_tensors.XS_test
    y_test = test_tensors.y_test
    s1_test = test_tensors.s1_test
    
    # Save return values here
    clean_result = []
    
    val = train_opt.val # Number of data points in validation set
    k = train_opt.k     # Update ratio of generator and discriminator (1:k training).
    n_epochs = train_opt.n_epochs  # Number of training epoch
    
    # Change the input validation data to an appropriate shape for the traininig
    XSY_val = torch.cat([XS_val, y_val.reshape((y_val.shape[0], 1))], dim=1)  

    # The loss values of each component will be saved in the following lists. 
    # We can draw epoch-loss graph by the following lists, if necessary.
    g_losses =[]
    d_f_losses = []
    d_r_losses = []

    BCE_loss = torch.nn.BCELoss()

    # Initialize a generator and two discriminators
    generator = Generator()
    discriminator_F = Discriminator_F() # Fairness discriminator
    discriminator_R = Discriminator_R() # Robustness discriminator

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator_F.apply(weights_init_normal)
    discriminator_R.apply(weights_init_normal)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=train_opt.lr_g)
    optimizer_D_F = torch.optim.SGD(discriminator_F.parameters(), lr=train_opt.lr_f)
    optimizer_D_R = torch.optim.SGD(discriminator_R.parameters(), lr=train_opt.lr_r)

    
    XSY_val_data = XSY_val[:val]

    train_len = XS_train.shape[0]
    val_len = XSY_val.shape[0]

    # Ground truths using in Disriminator_R
    Tensor = torch.FloatTensor
    valid = Variable(Tensor(train_len, 1).fill_(1.0), requires_grad=False)
    fake = Variable(Tensor(train_len, 1).fill_(0.0), requires_grad=False)
    clean = Variable(Tensor(val_len, 1).fill_(1.0), requires_grad=False)


    for epoch in range(n_epochs):

        # -----------------
        #  Forward Generator
        # -----------------
        if epoch % k == 0 or epoch < 300:
            optimizer_G.zero_grad()

        gen_y = generator(XS_train)
        gen_data = torch.cat([XS_train, gen_y.detach().reshape((train_len, 1))], dim=1)


        # ---------------------
        #  Train Fairness Discriminator
        # ---------------------

        optimizer_D_F.zero_grad()

        # Discriminator_F try to distinguish the sensitive groups by using the output of the generator.
        
        d_f_loss = BCE_loss(discriminator_F(gen_y.detach()), s1_train)

        d_f_loss.backward()
        d_f_losses.append(d_f_loss)
        optimizer_D_F.step()
        
        
        # ---------------------
        #  Train Robustness Discriminator
        # ---------------------
        optimizer_D_R.zero_grad()

        # Discriminator_R try to distinguish whether the input is from the validation data or the generated data from generator.
        
        clean_loss = BCE_loss(discriminator_R(XSY_val_data), clean)
        poison_loss = BCE_loss(discriminator_R(gen_data.detach()), fake)
        d_r_loss = 0.5 * (clean_loss + poison_loss)

        d_r_loss.backward()
        d_r_losses.append(d_r_loss)
        optimizer_D_R.step()
        

        # ---------------------
        #  Update Generator
        # ---------------------

        # Loss measures generator's ability to fool the discriminators

        if epoch < 300 : # This if-statement is for pre-training. This is optional.
            g_loss = (1-lambda_r) * BCE_loss((torch.tanh(gen_y)+1)/2, (y_train+1)/2) + lambda_r * BCE_loss(discriminator_R(gen_data), valid) 
            g_loss.backward()
            g_losses.append(g_loss)
            optimizer_G.step()

        elif epoch % k == 0:
            s_gen = BCE_loss(discriminator_F(gen_y), 1-s1_train)
            g_loss = (1-lambda_r-lambda_f) * BCE_loss((torch.tanh(gen_y)+1)/2, (y_train+1)/2) + lambda_r * BCE_loss(discriminator_R(gen_data), valid)  + lambda_f * s_gen
            g_loss.backward()

            optimizer_G.step()

        g_losses.append(g_loss)

        if epoch % 100 == 0:
            print(
                "[Lambda_f: %.2f] [Epoch %d/%d]"
                % (lambda_f, epoch, n_epochs),  end="\r"
            )

    tmp = test_model(generator, XS_test, y_test, s1_test)
    clean_result.append([lambda_f, tmp[0].item(), tmp[1]])
        
    return clean_result


'''
def main() : Load data and call other functions for training FR-GAN.
If the input argument is 'clean', then the main function will train FR-GAN with 'clean' synthetic data. (Default)
If the input argument is 'poison', then the main function will train FR-GAN with 'poisoned' synthetic data.
'''
def main(clean="clean"):
    
    # a namespace object which contains some of the dataset related hyperparameters
    data_opt = Namespace(num_train=2000, num_val1=200, num_val2=500, num_test=1000, num_plot=300)
    
    # Load synthetic dataset
    # These dataset is generated by the open sourced code (Zafar et al., 2017) as we described in the paper.
    X = np.load('X_synthetic.npy')
    y = np.load('y_synthetic.npy')
    s1 = np.load('s1_synthetic.npy')
    
    # Split the raw data. The returned values are tuples of torch.FloatTensors
    (XS_train, y_train, s1_train), (XS_val, y_val, s1_val), (XS_test, y_test, s1_test) = arrange_data(X, y, s1, data_opt)

    # Define namespaces. These are the input arguments of train_model(). 
    # Under poisoning set-up, train_tensors will be replaces by poi_tensors.)
    train_tensors = Namespace(XS_train = XS_train, y_train = y_train, s1_train = s1_train)
    val_tensors = Namespace(XS_val = XS_val, y_val = y_val, s1_val = s1_val) 
    test_tensors = Namespace(XS_test = XS_test, y_test = y_test, s1_test = s1_test)
    
    train_result = []
    
    if clean == "poison":
        
        # Change the clean data to the poisoned data
        y_poi, s1_poi = get_poisoned_data(y_train, s1_train, poi_type = 's', poi_ratio = 0.1)
        XS_poi = torch.cat([XS_train[:, :2].detach(), s1_poi.detach().reshape((s1_poi.shape[0], 1))], dim=1)       
        poi_tensors = Namespace(XS_train = XS_poi, y_train = y_poi, s1_train = s1_poi) # Define a namespace for the poisoned data tentors
        
        train_opt = Namespace(val=len(y_val), n_epochs=3000, k=3, lr_g=0.01, lr_f=0.01, lr_r=0.01)
        lambda_f_set = [0.1,0.2,0.25,0.3,0.35,0.4,0.45,0.5] # Various lambda values for the fairness discriminator of FR-GAN
        lambda_r = 0.2 # Lambda value for the robustness discriminator of FR-GAN
        
        for lambda_f in lambda_f_set:
            train_result.append(train_model(poi_tensors, val_tensors, test_tensors, train_opt, lambda_f = lambda_f, lambda_r = lambda_r))
    else:
        
        train_opt = Namespace(val=len(y_val), n_epochs=4000, k=3, lr_g=0.005, lr_f=0.005, lr_r=0.0001)        
        lambda_f_set = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95] # Various lambda values for the fairness discriminator of FR-GAN
        lambda_r = 0.01 # Lambda value for the robustness discriminator of FR-GAN.
        for lambda_f in lambda_f_set:
            train_result.append(train_model(train_tensors, val_tensors, test_tensors, train_opt, lambda_f = lambda_f, lambda_r = lambda_r))
    
    print("----------------------------------------------------------")
    print("-------- Training Results of FR-GAN on %s data --------" %clean)
    for i in range(len(train_result)):
        print(
            "[Lambda: %.2f] Accuracy : %.3f, Disparate Impact : %.3f "
            % (train_result[i][0][0], train_result[i][0][1], train_result[i][0][2])
        )       
    print("----------------------------------------------------------")
    
if __name__ == '__main__':
    clean = 'clean'
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main(clean)
