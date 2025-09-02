import h5py as h5
import numpy as np

def load_contribution_data(file_path, data_type, layer=0, sign='concat', norm=True):

    layer = str(layer)  # Ensure layer is a string

    file = h5.File(file_path, 'r')
    targets= file['targets'][:]
    data = file[layer][data_type]

    if sign == 'concat':
        negative = data['negative'][:]
        positive = data['positive'][:]

        data = np.concatenate((positive, negative), axis=1)

    elif sign == 'sum':
        data = data['negative'][:] + data['positive'][:]
        

    elif sign == 'positive':
        data = data['positive'][:]

    elif sign == 'negative':
        data = data['negative'][:]
    
    if norm:
        data = data / np.std(data)

    return data, targets

