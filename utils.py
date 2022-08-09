
import sys
import librosa
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import pickle

########################
# TRANSFORM DATA TOOLS #
########################


def spec_to_img(spec, eps=1e-6):

    '''
    transform spectrum data into an image
    '''
    # set the mean and std of the spectrum decible data
    mean = spec.mean()
    std = spec.std()
    # normalize the spectrum data with the mean and std
    spec_norm = (spec - mean) / (std + eps)
    # fetch the min and max of the spectrum data
    spec_min, spec_max = spec_norm.min(), spec_norm.max()
    # scale the spectrum data to 255 in order to be read as image data
    spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
    spec_scaled = spec_scaled.astype(np.uint8)

    return spec_scaled


def melspectrogram_db(fpath, sr=None, n_fft=2048, hop_length=512, n_mels=128, fmin=20, fmax=8300, top_db=80):

    '''
    extract mel spectrum decible data from audio
    '''
    # load wav file in using librosa
    wav, sr = librosa.load(fpath, sr=sr)
    # if file is less than 5secs pad the data to fit 5secs
    if wav.shape[0]<5*sr:

        wav = np.pad(wav, int(np.ceil((5*sr-wav.shape[0])/2)), mode='reflect')
    else:
        # if the file is greater than or equal to 5secs, trim it to 5secs
        wav=wav[:5*sr]
    # fetch melspectrogram using librosa, then convert the spectrum to decibles
    spec = librosa.feature.melspectrogram(wav, sr=sr, n_fft=n_fft,
                                         hop_length=hop_length, n_mels=n_mels,
                                         fmin=fmin, fmax=fmax)
    spec_db = librosa.power_to_db(spec, top_db=top_db)

    return spec_db


#############################
# NEURAL NET TRAINING TOOLS #
#############################

def set_learning_rate(optimizer, lr):

    '''
    set learning rate to optimizer's parameters
    '''

    for param_group in optimizer.param_groups:

        param_group['lr'] = lr
    
    return optimizer

def learning_rate_decay(optimizer, epoch, learning_rate):

    '''
    set decay to learning rate every 20th epoch
    '''

    if epoch % 20 == 0:

        lr = learning_rate / (100**(epoch//20))
        opt = set_learning_rate(optimizer, lr)
        print(f'[+] Changing Learning Rate to: {lr}')

        return opt
    
    else:

        return optimizer


def train(model, train_loader, valid_loader, epochs=100, learning_rate=2e-5, decay=True):

    '''
    training loop for model
    '''
    # check GPU availabilty
    if torch.cuda.is_available():

        device=torch.device('cuda:0')
    else:

        device=torch.device('cpu')
    # set necessary variables for training
    loss_func = nn.CrossEntropyLoss()
    learning_rate = learning_rate
    opt = optim.Adam(model.parameters(), lr=learning_rate)
    epochs = epochs

    train_losses = []
    valid_losses = []

    for e in range(1, epochs + 1):

        # assigning model to training
        model.train()

        batch_losses = []

        if decay:
            opt = learning_rate_decay(opt, e, learning_rate)
        
        zx = 0
        for i, data in enumerate(train_loader):
            zx += 1
            x,y = data
            # zero out gradients
            opt.zero_grad()
            # set data batch to GPU if available
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.long)
            zf = ('[i] FORWARD' + '-' * (zx//4) +'> * <'+ '-'* (zx//4) + 'BACKWARD [i] ')
            # forward pass
            preds = model(x)
            # calculate loss
            loss = loss_func(preds, y)
            # backward pass
            loss.backward()
            # collect the loss per batch
            batch_losses.append(loss.item())
            # optimize gradients
            opt.step()
            sys.stdout.write('\r'+zf)
        # use collected batch loss to track training loss
        train_losses.append(batch_losses)
        print(f'Epoch - {e} Train-Loss: {np.mean(train_losses[-1])}')

        # set model to evaluate
        model.eval()

        batch_losses = []
        trace_y = []
        trace_pred = []

        zx = 0
        for i, data in enumerate(valid_loader):
            zx += 1
            x, y = data 
            # set data to GPU
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.long)
            zf = ('[i] FORWARD' + '-' * (zx//4) + '> *  ')
            # forward pass
            preds = model(x)
            # calculate loss
            loss = loss_func(preds, y)
            # collect loss data to generate current model accuracy
            trace_y.append(y.cpu().detach().numpy())
            trace_pred.append(preds.cpu().detach().numpy())
            batch_losses.append(loss.item())
            sys.stdout.write('\r'+zf)
        valid_losses.append(batch_losses)
        trace_y = np.concatenate(trace_y)
        trace_pred = np.concatenate(trace_pred)

        acc = np.mean(trace_pred.argmax(axis=1) == trace_y)

        print(f'Epoch - {e} Valid-Loss: {np.mean(valid_losses[-1])} Valid Accuracy: {acc}')
   
    return model

#################
# STORING TOOLS #
#################

def save_model(model, fname):

    '''
    save pytorch trained model
    '''

    with open(fname, 'wb') as f:
        torch.save(model, f)

def save_cat_idx(data, fname):

    '''
    save categorical look up tables
    '''

    with open(fname, 'wb') as f:
        pickle.dump(data.idx2cat, f)







