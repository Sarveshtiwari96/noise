
import torch
import utils

import pickle

########################
# GENERATE PREDICTIONS #
########################


def predict(wav, cats, model):

    '''
    Recives: 
    1. classification model
    2. categorical look up tabels
    3. input audio
    
    Output:
    1. model classification of given input audio
    '''
    # check GPU availability
    if torch.cuda.is_available():

        device = torch.device('cuda:0')
    else:
        device=torch.device('CPU')

    path = wav
    # receive input audio - transform to model input
    spec = utils.spec_to_img(utils.melspectrogram_db(path))
    spec_tensor = torch.from_numpy(spec).to(device, dtype=torch.float32)
    # forward pass
    preds = model.forward(spec_tensor.reshape(1,1,*spec_tensor.shape))
    # isolate predicted index
    idx = preds.argmax(dim=1).cpu().detach().numpy().ravel()[0]
    # print index in categorical look up table to retrieve prediction
    print(f'This Audio file is classified as: {cats[idx]}')

    return cats[idx]



if __name__ == '__main__':

    ####################
    # TEST BOTH MODELS #
    ####################

    # set categorical look ups
    with open('models/idx2cat.pkl', 'rb') as f:
        cats = pickle.load(f)
    # set sameple audio
    p1 = '/content/drive/MyDrive/ESC-50/audio/1-100032-A-0.wav' # vacuum cleaner
    p2 = '/content/drive/MyDrive/ESC-50/audio/1-100038-A-14.wav' # dog
    p3 = '/content/drive/MyDrive/ESC-50/audio/1-100210-A-36.wav' # cat

    wavs = [p1, p2, p3]

    #################################
    # TEST CONVOLUTIONAL NEURAL NET #
    #################################

    CNN_model = torch.load('/content/SER-ESC-50/models/CNN_Classifier.pth')
    
    print('[i] Custom Predictions:')
    for wav in wavs:
        print(f'Prediction from file: {wav}')
        predict(wav=wav,
                cats=cats,
                model=CNN_model)

    ##################
    # TEST RESNET 34 #
    ##################

    res_model = torch.load('/content/SER-ESC-50/models/resNet.pth')

    print('[i] ResNET Predicitons:')

    for wav in wavs:
        print(f'Prediction from file: {wav}')
        predict(wav=wav,
                cats=cats,
                model=res_model)

    

    

