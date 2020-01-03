import numpy as np
import torch
import torch.nn.functional as F
import math
from argparse import Namespace

def to_torch_FT(X, y, s1):
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(s1, np.ndarray)
    return torch.FloatTensor(X), torch.FloatTensor(y), torch.FloatTensor(s1)

'''
def arrange_data() : Split the input raw data into three datasets (train set, validation set, and test set) and return them.
The amount of each dataset is defined by the given opt argument.
We currently do not use second validation set (which can be specified by opt.num_val2).
'''
def arrange_data(X, y, s1, opt):
    
    num_train = opt.num_train
    num_val1 = opt.num_val1
    num_val2 = opt.num_val2
    num_test = opt.num_test

    X, y, s1 = to_torch_FT(X, y, s1)
    
    X_train = X[:num_train-num_val1]
    y_train = y[:num_train-num_val1]
    s1_train = s1[:num_train-num_val1]

    X_val = X[num_train: num_train + num_val1]
    y_val = y[num_train: num_train + num_val1]
    s1_val = s1[num_train: num_train + num_val1]

    X_test = X[num_train + num_val1 + num_val2:num_train + num_val1 + num_val2 + num_test]
    y_test = y[num_train + num_val1 + num_val2:num_train + num_val1 + num_val2 + num_test]
    s1_test = s1[num_train + num_val1 + num_val2:num_train + num_val1 + num_val2 + num_test]
   
    XS_train = torch.cat([X_train, s1_train.reshape((s1_train.shape[0], 1))], dim=1)
    XS_val = torch.cat([X_val, s1_val.reshape((s1_val.shape[0], 1))], dim=1)
    XS_test = torch.cat([X_test, s1_test.reshape((s1_test.shape[0], 1))], dim=1)
 
    return (XS_train, y_train, s1_train), (XS_val, y_val, s1_val), (XS_test, y_test, s1_test)

'''
def get_poisoned_data() : Change the clean data to the poisoned data based on the poi_type and poi_ratio.
If the poisoned type is 's', then the s-flip (same as z-flip) poisoning will be done.
If the poisoned type is 'y', then the y-flip poisoning will be done.
'''
def get_poisoned_data(y_train, s1_train, poi_type='s', poi_ratio=0.1):
    
    # Get the flipped indexes from the external file. 
    # The indexes are chosed by the poisoning algorithm, which is described in the paper.
    flipped_idx = np.loadtxt("Synthetic_z_flip_poisoning.txt", dtype=np.int64) 
    n_idx = len(s1_train)*poi_ratio
    flipped_idx = flipped_idx[:int(n_idx)]
    
    if poi_type == 's':    
        s1_poi = np.copy(s1_train) 
        s1_poi[flipped_idx] = 1- s1_poi[flipped_idx]
        s1_poi = torch.FloatTensor(s1_poi)
        y_poi = np.copy(y_train)
        y_poi = torch.FloatTensor(y_poi)
        
    elif poi_type == 'y':
        y_poi = np.copy(y_train)
        y_poi[flipped_idx] = 0 - y_poi[flipped_idx]
        y_poi = torch.FloatTensor(y_poi)
        s1_poi = np.copy(s1_train)
        s1_poi = torch.FloatTensor(s1_poi)
            
    return y_poi, s1_poi


'''
def test_model() : Evaluate the test performance {accuracy and fairness (especially disparate impact)} of the model.
A model and three tensors for test (X, y, s1) should be given for the test.
The return values are the test accuracy and the value of disparate impact.
'''
def test_model(model_, X, y, s1):
    model_.eval()
    
    y_hat = model_(X).squeeze()

    prediction = (y_hat > 0.0).int().squeeze()
    y = (y > 0.0).int()

    z_0_mask = (s1 == 0.0)
    z_1_mask = (s1 == 1.0)
    z_0 = int(torch.sum(z_0_mask))
    z_1 = int(torch.sum(z_1_mask))

    Pr_y_hat_1_z_0 = float(torch.sum((prediction == 1)[z_0_mask])) / z_0
    Pr_y_hat_1_z_1 = float(torch.sum((prediction == 1)[z_1_mask])) / z_1
    
    y_1_z_0_mask = (y == 1.0) & (s1 == 0.0)
    y_1_z_1_mask = (y == 1.0) & (s1 == 1.0)
    y_1_z_0 = int(torch.sum(y_1_z_0_mask))
    y_1_z_1 = int(torch.sum(y_1_z_1_mask))

    Pr_y_hat_1_y_1_z_0 = float(torch.sum((prediction == 1)[y_1_z_0_mask])) / y_1_z_0
    Pr_y_hat_1_y_1_z_1 = float(torch.sum((prediction == 1)[y_1_z_1_mask])) / y_1_z_1
    
    y_hat_neq_y = float(torch.sum((prediction == y.int())))

    test_acc = torch.sum(prediction == y.int()).float() / len(y)
    
    min_dp = min(Pr_y_hat_1_z_0, Pr_y_hat_1_z_1)
    max_dp = max(Pr_y_hat_1_z_0, Pr_y_hat_1_z_1)
    
    print("Test accuracy: {}".format(test_acc))
    print("P(y_hat=1 | z=0) = {:.3f}, P(y_hat=1 | z=1) = {:.3f}".format(Pr_y_hat_1_z_0, Pr_y_hat_1_z_1))
    print("Disparate Impact ratio = {:.3f}".format(min_dp/max_dp))
    
#     min_eo = min(Pr_y_hat_1_y_1_z_0, Pr_y_hat_1_y_1_z_1)
#     max_eo = max(Pr_y_hat_1_y_1_z_0, Pr_y_hat_1_y_1_z_1)
#     print("Equal Opportunity ratio = {:.3f}".format(min_eo/max_eo))

    return test_acc, min_dp/max_dp
