import pickle
import numpy as np
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
from copy import deepcopy

class DataLoader(object):
    def __init__(self, xs, ys, xadj, yadj, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size

            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xadj_padding = np.repeat(xadj[-1:], num_padding, axis=0)
            yadj_padding = np.repeat(yadj[-1:], num_padding, axis=0)

            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)

            xadj = np.concatenate([xadj, xadj_padding], axis=0)
            yadj = np.concatenate([yadj, yadj_padding], axis=0)

        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys
        self.xadj = xadj
        self.yadj = yadj

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys, xadj, yadj = self.xs[permutation], self.ys[permutation],  self.xadj[permutation], self.yadj[permutation]
        self.xs = xs
        self.ys = ys
        self.xadj = xadj
        self.yadj = yadj

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))

                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                xadj_i = self.xadj[start_ind: end_ind, ...]
                yadj_i = self.yadj[start_ind: end_ind, ...]

                yield (x_i, y_i, xadj_i, yadj_i)
                self.current_ind += 1

        return _wrapper()


class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std, device):
        self.mean = mean
        self.std = std
        self.device = device

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * torch.Tensor(self.std) + torch.Tensor(self.mean)).to(self.device)


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def load_adj(pkl_filename, adjtype):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    if adjtype == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adjtype == "normlap":
        adj = [calculate_normalized_laplacian(adj_mx).astype(np.float32).todense()]
    elif adjtype == "symnadj":
        adj = [sym_adj(adj_mx)]
    elif adjtype == "transition":
        adj = [asym_adj(adj_mx)]
    elif adjtype == "doubletransition":
        adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
    elif adjtype == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    else:
        error = 0
        assert error, "adj type not defined"
    return sensor_ids, sensor_id_to_ind, adj

def binary(x, bits):
    mask = 2**torch.arange(bits).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()

def mask(a):
    shape = a.shape
    b = np.zeros(shape)
    a = a.reshape(a.shape[0], -1)
    b = b.reshape(b.shape[0], -1)

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            temp_v = a[i][j]
            if temp_v == 0:
                b[i][j] = 0 + 2 * j
            else:
                b[i][j] = 1 + 2 * j

    b = b.reshape(shape).astype(int)
    return b


def load_dataset(dataset_dir, batch_size, valid_batch_size= None, test_batch_size=None, device = "cpu"):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
        data['xadj_' + category] = cat_data['x_adj']
        data['yadj_' + category] = cat_data['y_adj']

    num_node = data['x_train'].shape[2]

    mean_train = data['x_train'][..., 0].reshape(-1, num_node).mean(axis=0)
    std_train = data['x_train'][..., 0].reshape(-1, num_node).std(axis=0)

    scaler = StandardScaler(mean=mean_train, std=std_train, device=device)
    data_copy = deepcopy(data)
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])

        data_copy['x_' + category] = binary(torch.tensor(np.round(data_copy['x_' + category][:, :, :, 0]).astype('int')).transpose(1, 2), 32).numpy()
        data_copy['x_' + category] = mask(data_copy['x_' + category])

        data['x_' + category] = np.concatenate((data['x_' + category], data_copy['x_' + category].swapaxes(1, 2)), axis=3)

    data['train_loader'] = DataLoader(data['x_train'], data['y_train'],data['xadj_train'], data['yadj_train'], batch_size)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'],data['xadj_val'], data['yadj_val'], valid_batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'],data['xadj_test'], data['yadj_test'], test_batch_size)
    data['scaler'] = scaler
    return data

def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_idmape(preds, labels, null_val=np.nan):
    y_true = labels.flatten()
    y_pred = preds.flatten()
    mask = [y_true != null_val]
    loss = torch.abs(y_pred[tuple(mask)]-y_true[tuple(mask)]).sum() / torch.abs(y_true[tuple(mask)]).sum()
    return loss


def metric(pred, real):
    mae = masked_mae(pred,real,0.0).item()
    mape = masked_mape(pred,real,0.0).item()
    rmse = masked_rmse(pred,real,0.0).item()
    idmape = masked_idmape(pred,real,0.0).item()
    return mae, mape, rmse, idmape

