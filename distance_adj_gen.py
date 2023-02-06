#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math

import numpy as np
import pandas as pd
import time
import torch
def gen_data(data, ntr, N):
    '''
    if flag:
        data=pd.read_csv(fname)
    else:
        data=pd.read_csv(fname,header=None)
    '''
    #data=data.as_matrix()
    data=np.reshape(data,[-1,288,N])
    return data[0:ntr]

def normalize(a):

    # mu=np.mean(a,axis=1,keepdims=True)
    # std=np.std(a,axis=1,keepdims=True)

    mu = np.mean(a)
    std = np.std(a)

    return (a-mu)/std

def compute_dtw_version0(a,b,o=1,T=12):
    a=normalize(a)
    b=normalize(b)
    d=np.reshape(a,[1,1,T0])-np.reshape(b,[1,T0,1])
    d=np.linalg.norm(d,axis=0,ord=o)
    D=np.zeros([T0,T0])
    for i in range(T0):
        for j in range(max(0,i-T),min(T0,i+T+1)):
            if (i==0) and (j==0):
                D[i,j]=d[i,j]**o
                continue
            if (i==0):
                D[i,j]=d[i,j]**o+D[i,j-1]
                continue
            if (j==0):
                D[i,j]=d[i,j]**o+D[i-1,j]
                continue
            if (j==i-T):
                D[i,j]=d[i,j]**o+min(D[i-1,j-1],D[i-1,j])
                continue
            if (j==i+T):
                D[i,j]=d[i,j]**o+min(D[i-1,j-1],D[i,j-1])
                continue
            D[i,j]=d[i,j]**o+min(D[i-1,j-1],D[i-1,j],D[i,j-1])
    return D[-1,-1]**(1.0/o)


def compute_dtw_version1(a,b,o=2):
    a = normalize(a)
    b = normalize(b)
    d=np.reshape(a,[1,1,len(a)])-np.reshape(b,[1,len(b),1])
    d=np.linalg.norm(d,axis=0,ord=o)
    D=np.zeros([len(a),len(b)])
    for i in range(len(a)):
        for j in range(len(b)):
            if (i==0) and (j==0):
                D[i,j]=d[i,j]**o
                continue
            if (i==0):
                D[i,j]=d[i,j]**o+D[i,j-1]
                continue
            if (j==0):
                D[i,j]=d[i,j]**o+D[i-1,j]
                continue
            D[i,j]=d[i,j]**o+min(D[i-1,j-1],D[i-1,j],D[i,j-1])

    s = D[-1,-1]**(1.0/o)
    paths = np.sqrt(D)
    return s, paths.T

def best_path(paths):
    """ Compute the optimal path by backtrack"""
    i, j = int(paths.shape[0]-1), int(paths.shape[1]-1)
    p =[]
    if paths[i,j] !=-1:
       p.append((i-1, j-1))
    while i >0 and j>0:
        c = np.argmin([paths[i-1, j-1], paths[i-1, j], paths[i, j-1]])
        if c ==0:
            i, j = i-1, j-1
        elif c==1:
            i = i -1
        elif c==2:
            j = j-1
        if paths[i,j] != -1:
            p.append((i-1, j-1))
    p.pop()
    p.reverse()
    return p

def get_common_seq(best_path, threshold =1):
    com_ls = []
    pre = best_path[0]
    length =1
    for i, element in enumerate(best_path):
        if i ==0:
            continue
        cur = best_path[i]
        if cur[0] == pre[0] +1 and cur[1] == pre[1] +1 :
            length = length +1
        else:
            com_ls.append(length)
            length = 1
        pre = cur
    com_ls.append(length)
    return list(filter(lambda num: True if threshold< num else False, com_ls))


def calcuate_align_weight(seq_len, com_ls):
    weight = 0
    for com_len in com_ls:
        weight = weight + (com_len * com_len) / (seq_len * seq_len)
    return 1 - math.sqrt(weight)


def transport_adj_01(d, adj_percent):
    adj = d
    adj = adj + adj.T
    N = d.shape[0]

    w_adj = np.zeros([N, N])
    top = int(N * adj_percent)

    for i in range(adj.shape[0]):
        a = adj[i, :].argsort()[0:top]
        for j in range(top):
            w_adj[i, a[j]] = 1

    for i in range(N):
        for j in range(N):
            if (w_adj[i][j] != w_adj[j][i] and w_adj[i][j] == 0):
                w_adj[i][j] = 1
            if (i == j):
                w_adj[i][j] = 1

    print("Total route number: ", N)
    print(len(w_adj.nonzero()[0]) / (N * N))

    return w_adj


def compute_dtw_weighted(a,b,o=2):
    s_dtw, paths = compute_dtw_version1(a,b,o=2)
    p = best_path(paths)
    com_ls = get_common_seq(p)
    weight = calcuate_align_weight(len(p), com_ls)
    s_dtw_weighted = s_dtw * np.exp(weight)

    return s_dtw, s_dtw_weighted


if __name__ == '__main__':

    data = np.load('./data/rawdata/volume.npy')
    _ , sort_num = data.shape
    d1 = np.zeros([sort_num, sort_num])   ## dtw
    d2 = np.zeros([sort_num, sort_num])   ## dtw_weighted
    T0 = data.shape[1]

    for i in range(sort_num):
        t1=time.time()
        for j in range(i+1,sort_num):
            d1[i,j], d2[i,j] = compute_dtw_weighted(data[:, i], data[:, j])
        t2=time.time()
        print(t2-t1)
        print("=======================")

    dir_path = './data/TH/'
    np.save(dir_path + "dtw_weighted_TH_adj.npy", d2 + d2.T)
    print("The weighted matrix of temporal graph is generated!")


