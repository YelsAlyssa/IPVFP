import torch
import numpy as np
import argparse
import time
import new_util
import matplotlib.pyplot as plt
from model import *

parser = argparse.ArgumentParser()
# parser.add_argument('--device',type=str,default='cuda:3',help='')
parser.add_argument('--device',type=str,default='cpu',help='')
parser.add_argument('--data',type=str,default='data/TH',help='data path')
parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')

parser.add_argument('--gcn_bool',action='store_false',help='whether to add graph convolution layer')
parser.add_argument('--aptonly',action='store_true',help='whether only adaptive adj')
parser.add_argument('--addaptadj',action='store_false',help='whether add adaptive adj')

parser.add_argument('--seq_x',type=int,default=14,help='')
parser.add_argument('--seq_y',type=int,default=7,help='')

parser.add_argument('--nhid',type=int,default=32,help='')
parser.add_argument('--in_dim',type=int,default=2,help='inputs dimension')
parser.add_argument('--batch_size',type=int,default=2,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--epochs',type=int,default=40,help='')
parser.add_argument('--print_every',type=int,default=50,help='')
parser.add_argument('--seed',type=int,default=0,help='random seed')
parser.add_argument('--save',type=str,default='./data/model/',help='save path')
parser.add_argument('--expid',type=int,default=0,help='experiment id')

args = parser.parse_args()

def main():

    #set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    dtw_adj = np.load('./New_data/TH/dtw_weighted_TH_adj_v1.npy')
    supports = [torch.tensor(dtw_adj).to(device).float()]

    if args.aptonly:
       supports = None


    dataloader = new_util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size, device)
    scaler = dataloader['scaler']

    engine = trainer(scaler, args.in_dim, args.seq_x, args.seq_y, args.nhid, args.dropout, args.learning_rate,
                     args.weight_decay, device, supports, args.gcn_bool, args.addaptadj)

    print("start training...", flush=True)
    his_loss = []
    val_time = []
    train_time = []
    for i in range(1, args.epochs + 1):
        # if i % 10 == 0:
        # lr = max(0.000002,args.learning_rate * (0.1 ** (i // 10)))
        # for g in engine.optimizer.param_groups:
        # g['lr'] = lr
        train_loss = []
        train_mape = []
        train_rmse = []
        train_idmape = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y, x_adj, y_adj) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            trainx = trainx.transpose(1, 3)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)
            x_adj = torch.Tensor(x_adj).to(device)
            metrics = engine.train(trainx, trainy[:, 0, :, :], x_adj)
            train_loss.append(metrics[0])     ## mae
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            train_idmape.append(metrics[3])
            if iter % args.print_every == 0:
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Train IDMAPE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1], train_idmape[-1]), flush=True)
        t2 = time.time()
        train_time.append(t2 - t1)
        # validation
        valid_loss = []
        valid_mape = []
        valid_rmse = []
        valid_idmape = []
        s1 = time.time()
        for iter, (x, y, x_adj, y_adj) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            x_adj = torch.Tensor(x_adj).to(device)
            metrics = engine.eval(testx, testy[:, 0, :, :], x_adj)
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
            valid_idmape.append(metrics[3])
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i, (s2 - s1)))
        val_time.append(s2 - s1)

        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)
        mtrain_idmape = np.mean(train_idmape)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        mvalid_idmape = np.mean(valid_idmape)

        his_loss.append(mvalid_loss)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Train IDMAPE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Valid IDMAPE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mtrain_idmape, mvalid_loss, mvalid_mape, mvalid_rmse,mvalid_idmape, (t2 - t1)),
              flush=True)
        torch.save(engine.model.state_dict(),
                   args.save + "_epoch_" + str(i) + "_" + str(round(mvalid_loss, 2)) + ".pth")
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    # testing
    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(
        torch.load(args.save + "_epoch_" + str(bestid + 1) + "_" + str(round(his_loss[bestid], 2)) + ".pth"))

    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]

    for iter, (x, y, x_adj, y_adj) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        x_adj = torch.Tensor(x_adj).to(device)
        with torch.no_grad():
            preds = engine.model(testx, x_adj).transpose(1, 3).squeeze(1)
        outputs.append(preds)

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]

    print("Training finished")
    print("The valid loss on best model is", str(round(his_loss[bestid], 4)))

    amae = []
    amape = []
    armse = []
    aidmape = []
    for i in range(args.seq_y):
        pred = scaler.inverse_transform(yhat[:, :, i])
        real = realy[:, :, i]
        metrics = new_util.metric(pred, real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f},Test RMSE: {:.4f}, Test IDMAPE: {:.4f}'
        print(log.format(i + 1, metrics[0], metrics[1], metrics[2], metrics[3]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])
        aidmape.append(metrics[3])


    log = 'On average over {:d} horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}, Test IDMAPE: {:.4f}'
    print(log.format(args.seq_y, np.mean(amae), np.mean(amape), np.mean(armse), np.mean(aidmape)))
    torch.save(engine.model.state_dict(),
               args.save + "_exp" + str(args.expid) + "_best_" + str(round(his_loss[bestid], 2)) + ".pth")


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))