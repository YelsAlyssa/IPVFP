import torch
import numpy as np
import pandas as pd
import datetime
import  os

def embedding_days(date):

    year = int(date[:4])
    month = int(date[4:6])
    day = int(date[-2:])
    weekday = datetime.date(year, month, day).isoweekday()

    month_in_year = 12
    days_in_month = 30
    days_in_week = 7

    embedding = []

    sin_month= np.sin(2 * np.pi * month / month_in_year)
    cos_month = np.cos(2 * np.pi * month / month_in_year)

    sin_day = np.sin(2 * np.pi * day / days_in_month)
    cos_day = np.cos(2 * np.pi * day / days_in_month)

    sin_week = np.sin(2 * np.pi * weekday / days_in_week)
    cos_week = np.cos(2 * np.pi * weekday / days_in_week)

    # embedding = [sin_month] + [cos_month] + [sin_day] + [cos_day] + [sin_week] + [cos_week]

    embedding = [sin_month] + [sin_week]

    embedding = np.array(embedding).reshape(1,2)

    return embedding
def generate_feature_data(graph_th):
    N = len(graph_th.keys())
    node_num = graph_th[sorted(graph_th.keys())[0]].y.shape[0]
    indim = 2
    channel_dim =11
    data = np.zeros((N, node_num, indim))
    data_adj = np.zeros((N, node_num, node_num,channel_dim))

    for k in range(N):
        date = str(sorted(graph_th.keys())[k])
        value = graph_th[sorted(graph_th.keys())[k]].y.numpy()

        edge_index = graph_th[sorted(graph_th.keys())[k]].edge_index
        edge_timing_distribution = graph_th[sorted(graph_th.keys())[k]].edge_timing_dstribution
        edge_volume_distribution = graph_th[sorted(graph_th.keys())[k]].edge_volume_distribution

        node_num = value.shape[0]
        week_days = np.tile(datetime.date(int(date[:4]), int(date[4:6]), int(date[-2:])).isoweekday(), [node_num, 1])

        time_step1 = 1
        time_step2 = 10

        D1 = np.zeros((node_num, node_num, time_step1))
        D2 = np.zeros((node_num, node_num, time_step2))

        for i in range(edge_index.shape[1]):
            D1[edge_index[0][i], edge_index[1][i], 0] = edge_volume_distribution[i][0]
            for j in range(time_step1):
                D2[edge_index[0][i], edge_index[1][i], j] = edge_timing_distribution[i][j]

        D = np.concatenate((D1, D2), axis=2)
        feature = np.concatenate((value, week_days), axis=1)
        data[k,:,:] = feature
        data_adj[k,:,:,:] = D

    return data, data_adj


def generate_feature_data_abnormal(graph_th,index_set):

    # data1 = np.array(graph_th[sorted(graph_th.keys())[-1]].x)
    # data2 = np.array(graph_th[sorted(graph_th.keys())[-1]].y)
    # data_volume = np.concatenate((data1, data2), axis=1)

    N = len(graph_th.keys())
    for n in range(N):
        if n ==0:
            data_volume = np.array(graph_th[sorted(graph_th.keys())[n]].y)
        else:
            temp_volume = np.array(graph_th[sorted(graph_th.keys())[n]].y)
            data_volume = np.concatenate((data_volume, temp_volume), axis=1)

    hub_nub = data_volume.shape[0]
    abnormal_ref = [int(np.percentile(data_volume[i,:], 50)/15) for i in range(hub_nub)]
    abnormal_ref = np.array(abnormal_ref).reshape(hub_nub,1)

    node_num = graph_th[sorted(graph_th.keys())[0]].y.shape[0]
    indim = 2 + 7
    channel_dim = 11
    node_num_new = len(index_set)
    data = np.zeros((N, node_num_new, indim))
    data_adj = np.zeros((N, node_num_new, node_num_new, channel_dim))

    for k in range(N):
        date = str(sorted(graph_th.keys())[k])
        value = graph_th[sorted(graph_th.keys())[k]].y.numpy()
        mask = ((value - abnormal_ref) >0)
        value = value * mask

        if k < 7:
           history_value = value.repeat(7,1)
        else:
           history_value = graph_th[sorted(graph_th.keys())[k]].x.numpy()[:,-7:]

        edge_index = graph_th[sorted(graph_th.keys())[k]].edge_index
        edge_timing_distribution = graph_th[sorted(graph_th.keys())[k]].edge_timing_dstribution
        edge_volume_distribution = graph_th[sorted(graph_th.keys())[k]].edge_volume_distribution

        time_step1 = 1
        time_step2 = 10

        D1 = np.zeros((node_num, node_num, time_step1))
        D2 = np.zeros((node_num, node_num, time_step2))

        for i in range(edge_index.shape[1]):
            D1[edge_index[0][i], edge_index[1][i], 0] = edge_volume_distribution[i][0]
            for j in range(time_step1):
                D2[edge_index[0][i], edge_index[1][i], j] = edge_timing_distribution[i][j]

        D1= D1[index_set,:,:][:,index_set,:]
        D2= D2[index_set,:,:][:,index_set,:]

        value = value[index_set, :]
        history_value = history_value[index_set,:]
        year = int(date[:4])
        month = int(date[4:6])
        day = int(date[-2:])
        week_days = np.tile(datetime.date(year, month, day).isoweekday(),[node_num_new,1])

        # week_days = np.tile(embedding_days(date),[node_num_new,1])
        D = np.concatenate((D1, D2), axis=2)
        feature = np.concatenate((value, week_days,history_value), axis=1)
        data[k, :, :] = feature
        data_adj[k, :, :, :] = D

    return data, data_adj


def generate_graph_seq2seq_data(data, data_adj, x_offsets, y_offsets):
    num_samples = data.shape[0]
    x, y, x_adj, y_adj = [], [], [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))
    for t in range(min_t, max_t):
        x.append(data[t + x_offsets, ...])
        x_adj.append(data_adj[t + x_offsets, ...])
        y.append(data[t + y_offsets, ...])
        y_adj.append(data_adj[t + y_offsets, ...])
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    x_adj = np.stack(x_adj, axis=0)
    y_adj = np.stack(y_adj, axis=0)
    return x, y, x_adj, y_adj

def generate_train_val_test(feature_path, adj_path, data_dir, seq_length_x, seq_length_y):

    data, data_adj = np.load(feature_path), np.load(adj_path)

    print(data.shape, data_adj.shape)

    x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))
    y_offsets = np.sort(np.arange(1, (seq_length_y + 1), 1))
    x, y, x_adj, y_adj = generate_graph_seq2seq_data(data, data_adj, x_offsets, y_offsets)

    print(x.shape, y.shape, x_adj.shape, y_adj.shape)

    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train
    x_train, y_train, xadj_train, yadj_train = x[:num_train], y[:num_train], x_adj[:num_train], y_adj[:num_train]
    x_val, y_val, xadj_val, yadj_val = x[num_train : num_train + num_val], y[num_train : num_train + num_val],x_adj[num_train : num_train + num_val], y_adj[num_train : num_train + num_val]
    x_test, y_test, xadj_test, yadj_test = x[-num_test:], y[-num_test:], x_adj[-num_test:], y_adj[-num_test:]

    for cat in ["train", "val", "test"]:
        _x, _y, _xadj, _yadj = locals()["x_" + cat], locals()["y_" + cat],locals()["xadj_" + cat], locals()["yadj_" + cat]
        print(cat,"x: ", _x.shape, "y: ", _y.shape, "x_adj: ", _xadj.shape, "y_adj: ", _yadj.shape)
        np.savez_compressed(
            os.path.join(data_dir, f"{cat}.npz"),
            x = _x,
            y = _y,
            x_adj = _xadj,
            y_adj = _yadj,
            x_offsets = x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets = y_offsets.reshape(list(y_offsets.shape) + [1]),

        )

if __name__ == '__main__':

   feature_path = './data/rawdata/feature.npy'
   adj_path = './data/rawdata/s_t_adj.npy'
   data_dir_path = './data/TH/'

   generate_train_val_test(feature_path=feature_path, adj_path=adj_path, data_dir=data_dir_path, seq_length_x=14, seq_length_y=7)






