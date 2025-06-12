import h5py as h5

def load_contribution_data(file_path, data_type, layer=0, sign='concat'):
    file = h5.File(file_path, 'r')
    targets= file['targets'][:]
    data = file[layer][data_type]

    if sign == 'concat':
        negative = data['negative'][:]
        positive = data['positive'][:]

        data = np.concatenate((positive, positive), axis=0)

    elif sign == 'positive':
        data = data['positive'][:]

    elif sign == 'negative':
        data = data['negative'][:]

    return data, targets
