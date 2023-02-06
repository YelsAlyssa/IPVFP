import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from functools import partial
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import new_util
import torch.optim as optim


class trainer():
    def __init__(self, scaler, in_dim, seq_x, seq_y, nhid , dropout, lrate, wdecay, device, supports, gcn_bool, addaptadj):
        self.model = TTG(seq_x, seq_y, dropout, supports=supports, gcn_bool=gcn_bool, addaptadj=addaptadj, in_dim=in_dim, residual_channels=nhid, dilation_channels=nhid)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = new_util.masked_mae
        self.scaler = scaler
        self.clip = 5

    def train(self, input, real_val, x_adj):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input, x_adj)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output.transpose(1,3)[...,0])
        predict = predict.unsqueeze(3).transpose(1,3)
        loss = self.loss(predict, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mape = new_util.masked_mape(predict,real,0.0).item()
        rmse = new_util.masked_rmse(predict,real,0.0).item()
        idmape =new_util.masked_idmape(predict,real,0.0).item()

        return loss.item(),mape,rmse,idmape

    def eval(self, input, real_val, x_adj):
        self.model.eval()
        output = self.model(input, x_adj)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        # predict = self.scaler.inverse_transform(output)
        predict = self.scaler.inverse_transform(output.transpose(1, 3)[..., 0])
        predict = predict.unsqueeze(3).transpose(1, 3)
        loss = self.loss(predict, real, 0.0)
        mape = new_util.masked_mape(predict,real,0.0).item()
        rmse = new_util.masked_rmse(predict,real,0.0).item()
        idmape = new_util.masked_idmape(predict,real,0.0).item()
        return loss.item(),mape,rmse, idmape



class NBeatsBlock(nn.Module):
    """
    N-BEATS block which takes a basis function as an argument.
    """

    def __init__(self, x_t_n_inputs: int, theta_n_dim: int, basis: nn.Module, n_layers: int, theta_n_hidden: list,
                 batch_normalization: bool, dropout_prob: float, activation: str):
        """
        """
        super().__init__()

        theta_n_hidden = [x_t_n_inputs] + theta_n_hidden

        self.batch_normalization = batch_normalization
        self.dropout_prob = dropout_prob
        self.activations = {'relu': nn.ReLU(),
                            'softplus': nn.Softplus(),
                            'tanh': nn.Tanh(),
                            'selu': nn.SELU(),
                            'lrelu': nn.LeakyReLU(),
                            'prelu': nn.PReLU(),
                            'sigmoid': nn.Sigmoid()}

        hidden_layers = []
        for i in range(n_layers):

            # Batch norm after activation
            hidden_layers.append(nn.Linear(in_features=theta_n_hidden[i], out_features=theta_n_hidden[i + 1]))
            hidden_layers.append(self.activations[activation])

            if self.batch_normalization:
                hidden_layers.append(nn.BatchNorm1d(num_features=theta_n_hidden[i + 1]))

            if self.dropout_prob > 0:
                hidden_layers.append(nn.Dropout(p=self.dropout_prob))

        output_layer = [nn.Linear(in_features=theta_n_hidden[-1], out_features=theta_n_dim)]
        layers = hidden_layers + output_layer

        self.layers = nn.Sequential(*layers)
        self.basis = basis

    def forward(self, insample_y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        theta = self.layers(insample_y)
        backcast, forecast = self.basis(theta)
        return backcast, forecast


class IdentityBasis(nn.Module):
    def __init__(self, backcast_size: int, forecast_size: int):
        super().__init__()
        self.forecast_size = forecast_size
        self.backcast_size = backcast_size

    def forward(self, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        backcast = theta[:, :self.backcast_size]
        forecast = theta[:, -self.forecast_size:]
        return backcast, forecast


class TrendBasis(nn.Module):
    def __init__(self, degree_of_polynomial: int, backcast_size: int, forecast_size: int):
        super().__init__()
        polynomial_size = degree_of_polynomial + 1
        self.backcast_basis = nn.Parameter(
            torch.tensor(
                np.concatenate([np.power(np.arange(backcast_size, dtype=np.float32) / backcast_size, i)[None, :]
                                for i in range(polynomial_size)]), dtype=torch.float32), requires_grad=False)
        self.forecast_basis = nn.Parameter(
            torch.tensor(
                np.concatenate([np.power(np.arange(forecast_size, dtype=np.float32) / forecast_size, i)[None, :]
                                for i in range(polynomial_size)]), dtype=torch.float32), requires_grad=False)

    def forward(self, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cut_point = self.forecast_basis.shape[0]
        backcast = torch.einsum('bp,pt->bt', theta[:, cut_point:], self.backcast_basis)
        forecast = torch.einsum('bp,pt->bt', theta[:, :cut_point], self.forecast_basis)
        return backcast, forecast


class SeasonalityBasis(nn.Module):
    def __init__(self, harmonics: int, backcast_size: int, forecast_size: int):
        super().__init__()
        frequency = np.append(np.zeros(1, dtype=np.float32),
                              np.arange(harmonics, harmonics / 2 * forecast_size,
                                        dtype=np.float32) / harmonics)[None, :]
        backcast_grid = -2 * np.pi * (
                np.arange(backcast_size, dtype=np.float32)[:, None] / forecast_size) * frequency
        forecast_grid = 2 * np.pi * (
                np.arange(forecast_size, dtype=np.float32)[:, None] / forecast_size) * frequency

        backcast_cos_template = torch.tensor(np.transpose(np.cos(backcast_grid)), dtype=torch.float32)
        backcast_sin_template = torch.tensor(np.transpose(np.sin(backcast_grid)), dtype=torch.float32)
        backcast_template = torch.cat([backcast_cos_template, backcast_sin_template], dim=0)

        forecast_cos_template = torch.tensor(np.transpose(np.cos(forecast_grid)), dtype=torch.float32)
        forecast_sin_template = torch.tensor(np.transpose(np.sin(forecast_grid)), dtype=torch.float32)
        forecast_template = torch.cat([forecast_cos_template, forecast_sin_template], dim=0)

        self.backcast_basis = nn.Parameter(backcast_template, requires_grad=False)
        self.forecast_basis = nn.Parameter(forecast_template, requires_grad=False)

    def forward(self, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cut_point = self.forecast_basis.shape[0]
        backcast = torch.einsum('bp,pt->bt', theta[:, cut_point:], self.backcast_basis)
        forecast = torch.einsum('bp,pt->bt', theta[:, :cut_point], self.forecast_basis)
        return backcast, forecast


class NBeats(nn.Module):
    """
    N-Beats Model.
    """

    def __init__(self, blocks: nn.ModuleList):
        super().__init__()
        self.blocks = blocks

    def forward(self, insample_y: torch.Tensor, return_decomposition=False):

        residuals = insample_y.flip(dims=(-1,))
        forecast = insample_y[:, -1:]  # Level with Naive1

        block_forecasts = []
        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(insample_y=residuals)
            #residuals = residuals - backcast
            if i == 0:
                backcasts = backcast  # 要么forecast不加它自己
            else:
                backcasts += backcast
            forecast = forecast + block_forecast
            block_forecasts.append(block_forecast)

        block_forecasts = torch.stack(block_forecasts)
        block_forecasts = block_forecasts.permute(1, 0, 2)

        if return_decomposition:
            return forecast, backcasts, block_forecasts
        else:
            return forecast, backcasts

    def decomposed_prediction(self, insample_y: torch.Tensor):

        residuals = insample_y.flip(dims=(-1,))

        forecast = insample_y[:, -1:]  # Level with Naive1
        forecast_components = []
        for _, block in enumerate(self.blocks):
            backcast, block_forecast = block(residuals)
            residuals = residuals - backcast
            forecast = forecast + block_forecast
            forecast_components.append(block_forecast)
        return forecast, forecast_components


def init_weights(module, initialization):
    if type(module) == torch.nn.Linear:
        if initialization == 'orthogonal':
            torch.nn.init.orthogonal_(module.weight)
        elif initialization == 'he_uniform':
            torch.nn.init.kaiming_uniform_(module.weight)
        elif initialization == 'he_normal':
            torch.nn.init.kaiming_normal_(module.weight)
        elif initialization == 'glorot_uniform':
            torch.nn.init.xavier_uniform_(module.weight)
        elif initialization == 'glorot_normal':
            torch.nn.init.xavier_normal_(module.weight)
        elif initialization == 'lecun_normal':
            pass
        else:
            assert 1 < 0, f'Initialization {initialization} not found'


def create_stack_model(input_size: int, output_size: int, stack_types: list, n_blocks: list, batch_normalization: bool,
                       shared_weights: bool,
                       n_harmonics: int, n_layers: list, n_hidden: list, dropout_prob_theta: float, activation: str,
                       n_polynomials: int,
                       initialization: str):
    x_t_n_inputs = input_size

    if activation == 'selu': initialization = 'lecun_normal'

    # ------------------------ Model Definition ------------------------#
    block_list = []
    for i in range(len(stack_types)):
        for block_id in range(n_blocks[i]):

            # Batch norm only on first block
            if (len(block_list) == 0) and (batch_normalization):
                batch_normalization_block = True
            else:
                batch_normalization_block = False

            # Shared weights
            if shared_weights and block_id > 0:
                nbeats_block = block_list[-1]
            else:
                if stack_types[i] == 'seasonality':
                    nbeats_block = NBeatsBlock(x_t_n_inputs=x_t_n_inputs,
                                               theta_n_dim=4 * int(
                                                   np.ceil(n_harmonics / 2 * output_size) - (n_harmonics - 1)),
                                               basis=SeasonalityBasis(harmonics=n_harmonics,
                                                                      backcast_size=input_size,
                                                                      forecast_size=output_size),
                                               n_layers=n_layers[i],
                                               theta_n_hidden=n_hidden[i],
                                               batch_normalization=batch_normalization_block,
                                               dropout_prob=dropout_prob_theta,
                                               activation=activation)
                elif stack_types[i] == 'trend':
                    nbeats_block = NBeatsBlock(x_t_n_inputs=x_t_n_inputs,
                                               theta_n_dim=2 * (n_polynomials + 1),
                                               basis=TrendBasis(degree_of_polynomial=n_polynomials,
                                                                backcast_size=input_size,
                                                                forecast_size=output_size),
                                               n_layers=n_layers[i],
                                               theta_n_hidden=n_hidden[i],
                                               batch_normalization=batch_normalization_block,
                                               dropout_prob=dropout_prob_theta,
                                               activation=activation)
                elif stack_types[i] == 'identity':
                    nbeats_block = NBeatsBlock(x_t_n_inputs=x_t_n_inputs,
                                               theta_n_dim=input_size + output_size,
                                               basis=IdentityBasis(backcast_size=input_size,
                                                                   forecast_size=output_size),
                                               n_layers=n_layers[i],
                                               theta_n_hidden=n_hidden[i],
                                               batch_normalization=batch_normalization_block,
                                               dropout_prob=dropout_prob_theta,
                                               activation=activation)
                else:
                    assert 1 < 0, f'Block type not found!'
            # Select type of evaluation and apply it to all layers of block
            init_function = partial(init_weights, initialization=initialization)
            nbeats_block.layers.apply(init_function)
            block_list.append(nbeats_block)
    return NBeats(torch.nn.ModuleList(block_list))


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A)) ## matrix multiplication
        return x.contiguous() ## deep copy

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class Adj_apt(nn.Module):
    def __init__(self, c_in, c_out):
        super(Adj_apt, self).__init__()

        hidden_size_mlp = [128, 64]
        hidden_size_mlp = [308] + hidden_size_mlp
        layers = []
        for i in range(1, len(hidden_size_mlp)):
            layers.append(nn.Linear(in_features=hidden_size_mlp[i - 1], out_features=hidden_size_mlp[i]))
            layers.append(nn.ReLU())
            #layers.append(nn.BatchNorm1d(num_features=hidden_size_mlp[i]))
        layers = layers + [nn.Linear(in_features=hidden_size_mlp[-1], out_features=1)]
        layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_adj):
        adj = x_adj.reshape(-1, x_adj.shape[2], x_adj.shape[3], x_adj.shape[4])
        out = []
        for i in range(adj.shape[0]):
            adj_trans = torch.mul(adj[i, :, :, 0].unsqueeze(2), adj[i, :, :, 1:])
            adj_sets = torch.cat((adj[i, :, :, 0].unsqueeze(2), adj_trans), dim=2)
            out.append(adj_sets)
        h = torch.cat(out, dim=2)
        apt = self.mlp(h).squeeze(2)
        return apt



class TTG(nn.Module):
    def __init__(self, seq_x,
                       seq_y,
                       dropout=0.3,
                       supports=None,
                       gcn_bool=True,
                       addaptadj=True,
                       in_dim=2,
                       residual_channels=32,
                       dilation_channels=32,
                       blocks=4,
                       layers=2,
                       hidden_size_mlp1=[128,64],
                       hidden_size_mlp2 =[256, 64],
                       if_activate_last= False,
                       activation='relu',
                       batch_normalization=False,
                       dropout_prob=0
                       ):

        super(TTG, self).__init__()
        self.seq_x = seq_x
        self.seq_y = seq_y
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()


        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))

        self.conv2D_init_1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 2),
                                  dilation=2)

        self.conv2D_init_2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 3),
                                  dilation=2)
        self.conv2D_init_3 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 4),
                                  dilation=2)
        self.conv2D_init_4 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 5),
                                  dilation=2)


        self.q_conv = nn.Conv2d(in_channels=in_dim,
                                out_channels=residual_channels,
                                kernel_size=(1, 1))

        self.k_conv = nn.Conv2d(in_channels=in_dim,
                                out_channels=residual_channels,
                                kernel_size=(1, 1))

        self.v_conv = nn.Conv2d(in_channels=in_dim,
                                out_channels=residual_channels * 2,
                                kernel_size=(1, 1))

        self.supports = supports


        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)


        if self.gcn_bool:
            if self.addaptadj:
                self.gconv = gcn(dilation_channels,residual_channels,dropout,support_len=self.supports_len + 1)
            else:
                self.gconv = gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len)



        self.create_node_block_model = create_stack_model(input_size= self.seq_x,
                                            output_size= self.seq_y, stack_types=['seasonality'] + ['trend'] + ['identity'],
                                            n_blocks=[2, 2, 2], batch_normalization=False,
                                            shared_weights=False,
                                            n_harmonics=5,
                                            n_layers=[2, 2, 2], n_hidden=[[256, 256, 256], [128, 128, 128], [64, 64, 64]],
                                            dropout_prob_theta=0.005, activation='relu',
                                            n_polynomials=3,
                                            initialization='he_uniform')

        self.GRU = nn.GRU(1, 16, 2, batch_first=True)

        self.activations = {'relu': nn.ReLU(),
                            'softplus': nn.Softplus(),
                            'tanh': nn.Tanh(),
                            'selu': nn.SELU(),
                            'lrelu': nn.LeakyReLU(),
                            'prelu': nn.PReLU(),
                            'sigmoid': nn.Sigmoid()}

        hidden_size_mlp1 = [(12+10+8+6)*16] + hidden_size_mlp1
        layers1 = []
        for i in range(1, len(hidden_size_mlp1)):
            layers1.append(nn.Linear(in_features=hidden_size_mlp1[i - 1], out_features=hidden_size_mlp1[i]))
            layers1.append(self.activations[activation])
            if batch_normalization:
                layers1.append(nn.BatchNorm1d(num_features=hidden_size_mlp1[i]))
            if dropout_prob > 0:
                layers1.append(nn.Dropout(p=dropout_prob))
        layers1 = layers1 + [nn.Linear(in_features=hidden_size_mlp1[-1], out_features=self.seq_y)]
        if if_activate_last:
            layers1.append((self.activations[activation]))

        hidden_size_mlp2 = [self.seq_x] + hidden_size_mlp2
        layers2 = []
        for i in range(1, len(hidden_size_mlp2)):
            layers2.append(nn.Linear(in_features=hidden_size_mlp2[i - 1], out_features=hidden_size_mlp2[i]))
            layers2.append(self.activations[activation])
            if batch_normalization:
                layers2.append(nn.BatchNorm1d(num_features=hidden_size_mlp2[i]))
            if dropout_prob > 0:
                layers2.append(nn.Dropout(p=dropout_prob))
        layers2 = layers2 + [nn.Linear(in_features=hidden_size_mlp2[-1], out_features=self.seq_y)]
        if if_activate_last:
            layers2.append((self.activations[activation]))

        self.mlp1 = nn.Sequential(*layers1)

        self.mlp2 = nn.Sequential(*layers2)

        self.linear_q = nn.Linear(64, 32)

        self.linear_k = nn.Linear(64, 32)

        self.linear_v = nn.Linear(64, 32)

        self.embedding_week = nn.Embedding(8, 64)

        self.embedding_position = nn.Embedding(14, 64)

        self.binary_embedding = torch.nn.Embedding(73472, 64)

    def forward(self, input, x_adj):
        batch, feature_size,num_node, back_length = input.shape
        x = self.start_conv(input[:,:2,:,:])

        if self.supports is not None:
            self.supports[0] = F.softmax((F.relu(self.supports[0])), dim=1)

        new_supports = None
        if self.gcn_bool and self.addaptadj:
            apt = Adj_apt(c_in=x_adj.shape[0] * x_adj.shape[1] * 11, c_out=1)(x_adj)
            apt = F.softmax((F.relu(apt)), dim=1)
            if self.supports is not None:
                new_supports = self.supports + [apt]
            else:
                new_supports = [apt]

        x1 = self.conv2D_init_1(input[:,0,:,:].unsqueeze(1))
        x2 = self.conv2D_init_2(input[:,0,:,:].unsqueeze(1))
        x3 = self.conv2D_init_3(input[:,0,:,:].unsqueeze(1))
        x4 = self.conv2D_init_4(input[:,0,:,:].unsqueeze(1))

        x1 = x1.reshape(batch * num_node, x1.shape[3], -1)
        x2 = x2.reshape(batch * num_node, x2.shape[3], -1)
        x3 = x3.reshape(batch * num_node, x3.shape[3], -1)
        x4 = x4.reshape(batch * num_node, x4.shape[3], -1)

        out1, _ = self.GRU(x1)
        out2, _ = self.GRU(x2)
        out3, _ = self.GRU(x3)
        out4, _ = self.GRU(x4)
        out = torch.cat((out1, out2, out3, out4), dim=1)
        out = out.reshape(out.shape[0],-1)
        forecast1 = self.mlp1(out)
        forecast1 = forecast1.reshape(batch, num_node, -1)

        if self.gcn_bool:
            if self.addaptadj:
               x_gcn = self.gconv(x, new_supports)
            else:
               x_gcn = self.gconv(x, self.supports)
        x_gcn = torch.mean(x_gcn, dim=1)
        forecast2, _ = self.create_node_block_model(x_gcn.reshape(-1, x_gcn.shape[2]))
        forecast2 = forecast2.reshape(batch, num_node, -1)

        weekday = input[:, 1, :, :].int()
        position = torch.LongTensor([i for i in range(14)]).repeat(batch, num_node, 1)
        volume = input[:, 2:, :, :].reshape(batch, num_node, back_length, -1).int()
        volume_embedding = self.binary_embedding(volume)
        volume_embedding = torch.sum(volume_embedding, 3)
        volume_embedding = volume_embedding.reshape(-1,  volume_embedding.shape[2], 64)
        weekday = weekday.reshape(-1, weekday.shape[2])
        position = position.reshape(-1, position.shape[2])
        weekday_embedding = self.embedding_week(weekday)
        position_embedding = self.embedding_position(position)

        embedding = volume_embedding + weekday_embedding + position_embedding

        q = self.linear_q(embedding)
        k = self.linear_k(embedding)
        v = self.linear_v(embedding)

        attn = torch.matmul(q, k.permute(0, 2, 1))
        attn = torch.softmax(attn, dim=-1)
        attn_value = torch.matmul(attn, v)

        x_temp = torch.mean(attn_value, dim=2)
        x_temp = input[:, 0, :, :] + x_temp.reshape(batch, num_node, -1)
        forecast3 = self.mlp2(x_temp)

        forecast = forecast1 + forecast2 + forecast3

        forecast = forecast.transpose(1,2)

        return forecast.unsqueeze(3)



