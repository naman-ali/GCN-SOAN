import os.path as osp
import argparse
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
# from torch_geometric.nn import GCNSOANConv, ChebConv  # noqa
from modelsControllers.simulated_data.GCNSOANConv import GCNSOANConv
from torch_geometric.data import Data, DataLoader
import sys
import numpy as np
from requires import gcn_data_loader_no_val
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
from sklearn.model_selection import train_test_split
from requires.helper import get_experiments_ids
import os
import json
ROOT_DIR = os.getcwd() + '/'

experiment_id = None
folds = [1, 2, 3, 4]
ownership = True
friendship = True

test_size = 0.9
fold_results = []

embeddings = None
predicts = None

optuna_trials = 10

homophily_info = dict()


def prepare_data_for_gcn(experiment_id=None, fold=None):
    features, labels, G_df, folds = gcn_data_loader_no_val.get_simulated_data(experiment_id=experiment_id,
                                                                       ownership=ownership,
                                                                       test_size=test_size,
                                                                       friendship=friendship)

    x = torch.tensor(features, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.float)

    weights = torch.tensor(G_df['weight'].values, dtype=torch.float)

    edge_index = torch.tensor([G_df['source_id'].values,
                               G_df['target_id'].values], dtype=torch.long)

    data = Data(x=x, y=y, edge_index=edge_index, edge_attr=weights)

    train_idx = torch.from_numpy(folds[fold]['train'])
    test_idx = torch.from_numpy(folds[fold]['test'])

    data.train_mask = torch.tensor(np.full(labels.shape, False), dtype=torch.bool)
    data.train_mask[train_idx] = True


    # choose remained points as test
    data.test_mask = torch.tensor(np.full(labels.shape, False), dtype=torch.bool)
    # test_idx = ((~data.train_mask) & (data.y != -1)).nonzero(as_tuple=False).view(-1)
    data.test_mask[test_idx] = True

    assignment_idx = (data.y != -1).nonzero(as_tuple=False).view(-1)
    data.assign_mask = torch.tensor(np.full(labels.shape, False), dtype=torch.bool)
    data.assign_mask[assignment_idx] = True

    data.test_idx = test_idx

    return data, G_df


class Net(torch.nn.Module):

    def __init__(self, data, layer=None, params=None):
        super(Net, self).__init__()

        self.data = data
        self.layer = layer

        num_features = data.x.size(1)
        num_classes = torch.unique(data.y).size(0)


        if layer == 2:
            self.conv1 = GCNSOANConv(num_features, params['dimension'], cached=False,
                                 normalize=True)
            self.conv2 = GCNSOANConv(params['dimension'], 1, cached=False,
                                 normalize=True)

        if layer == 3:
            self.conv1 = GCNSOANConv(num_features, params['dimension'], cached=False,
                                 normalize=True)
            self.conv2 = GCNSOANConv(params['dimension'], params['dimension'], cached=False,
                                 normalize=True)
            self.conv3 = GCNSOANConv(params['dimension'], 1, cached=False,
                                 normalize=True)



    def forward(self):

        x, edge_index, edge_weight = self.data.x, self.data.edge_index, self.data.edge_attr

        if self.layer == 2:
            x = F.elu(self.conv1(x, edge_index, edge_weight))
            x = F.dropout(x, training=self.training, p=0.1)
            emb = x
            x = torch.sigmoid(self.conv2(x, edge_index, edge_weight))

        if self.layer == 3:
            x = F.elu(self.conv1(x, edge_index, edge_weight))
            x = F.dropout(x, training=self.training, p=0.0)
            x = F.elu(self.conv2(x, edge_index, edge_weight))
            emb = x
            x = F.elu(self.conv3(x, edge_index, edge_weight))

        
        return x, emb


def define_model(data, layer=None, params=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, data = Net(data, layer=layer, params=params).to(device), data.to(device)

    if layer == 1:
        optimizer = torch.optim.Adam([
            dict(params=model.conv1.parameters(), weight_decay=0)
        ], lr=params['learning_rate'])  # Only perform weight-decay on first convolution.

    if layer == 2:
        optimizer = torch.optim.Adam([
            dict(params=model.conv1.parameters(), weight_decay=0),
            dict(params=model.conv2.parameters(), weight_decay=0)
        ], lr=params['learning_rate'])  # Only perform weight-decay on first convolution.

    if layer == 3:
        optimizer = torch.optim.Adam([
            dict(params=model.conv1.parameters(), weight_decay=0),
            dict(params=model.conv2.parameters(), weight_decay=0),
            dict(params=model.conv3.parameters(), weight_decay=0)
        ], lr=params['learning_rate'])  # Only perform weight-decay on first convolution.

    if layer == 4:
        optimizer = torch.optim.Adam([
            dict(params=model.conv1.parameters(), weight_decay=0),
            dict(params=model.conv2.parameters(), weight_decay=0),
            dict(params=model.conv3.parameters(), weight_decay=0),
            dict(params=model.conv4.parameters(), weight_decay=0)
        ], lr=params['learning_rate'])  # Only perform weight-decay on first convolution.

    if layer == 5:
        optimizer = torch.optim.Adam([
            dict(params=model.conv1.parameters(), weight_decay=0),
            dict(params=model.conv2.parameters(), weight_decay=0),
            dict(params=model.conv3.parameters(), weight_decay=0),
            dict(params=model.conv4.parameters(), weight_decay=0),
            dict(params=model.conv5.parameters(), weight_decay=0)
        ], lr=params['learning_rate'])  # Only perform weight-decay on first convolution.

    if layer == 6:
        optimizer = torch.optim.Adam([
            dict(params=model.conv1.parameters(), weight_decay=0),
            dict(params=model.conv2.parameters(), weight_decay=0),
            dict(params=model.conv3.parameters(), weight_decay=0),
            dict(params=model.conv4.parameters(), weight_decay=0),
            dict(params=model.conv5.parameters(), weight_decay=0),
            dict(params=model.conv6.parameters(), weight_decay=0)
        ], lr=params['learning_rate'])  # Only perform weight-decay on first convolution.

    if layer == 7:
        optimizer = torch.optim.Adam([
            dict(params=model.conv1.parameters(), weight_decay=0),
            dict(params=model.conv2.parameters(), weight_decay=0),
            dict(params=model.conv3.parameters(), weight_decay=0),
            dict(params=model.conv4.parameters(), weight_decay=0),
            dict(params=model.conv5.parameters(), weight_decay=0),
            dict(params=model.conv6.parameters(), weight_decay=0),
            dict(params=model.conv7.parameters(), weight_decay=0)
        ], lr=params['learning_rate'])  # Only perform weight-decay on first convolution.

    if layer == 8:
        optimizer = torch.optim.Adam([
            dict(params=model.conv1.parameters(), weight_decay=0),
            dict(params=model.conv2.parameters(), weight_decay=0),
            dict(params=model.conv3.parameters(), weight_decay=0),
            dict(params=model.conv4.parameters(), weight_decay=0),
            dict(params=model.conv5.parameters(), weight_decay=0),
            dict(params=model.conv6.parameters(), weight_decay=0),
            dict(params=model.conv7.parameters(), weight_decay=0),
            dict(params=model.conv8.parameters(), weight_decay=0)
        ], lr=params['learning_rate'])  # Only perform weight-decay on first convolution.

    if layer == 9:
        optimizer = torch.optim.Adam([
            dict(params=model.conv1.parameters(), weight_decay=0),
            dict(params=model.conv2.parameters(), weight_decay=0),
            dict(params=model.conv3.parameters(), weight_decay=0),
            dict(params=model.conv4.parameters(), weight_decay=0),
            dict(params=model.conv5.parameters(), weight_decay=0),
            dict(params=model.conv6.parameters(), weight_decay=0),
            dict(params=model.conv7.parameters(), weight_decay=0),
            dict(params=model.conv8.parameters(), weight_decay=0),
            dict(params=model.conv9.parameters(), weight_decay=0)
        ], lr=params['learning_rate'])  # Only perform weight-decay on first convolution.

    return model, data, optimizer


def train(model=None, optimizer=None, data=None):
    model.train()
    optimizer.zero_grad()
    model_, emb = model()
    model_res = torch.reshape(model_[data.train_mask], (-1,))
    F.mse_loss(model_res, data.y[data.train_mask]).backward()
    optimizer.step()


@torch.no_grad()
def test(model=None, data=None):
    model.eval()
    predicts, emb = model()
    test_measures = []

    val_pred = pd.DataFrame()
    test_pred = pd.DataFrame()
    emb_all = np.ones((1, 32))
    node_tarcnets = np.ones((1, 1))
    for _, mask in data('train_mask', 'test_mask'):
        pred = torch.reshape(predicts[mask], (-1,))

        if _ == 'test_mask':
            test_pred['pred'] = pred.numpy()
            test_pred['ground'] = data.y[mask].numpy()
            test_pred['id'] = data.test_idx

        loss = F.mse_loss(pred, data.y[mask])

        test_measures.append(round(loss.item(), 20))

    # test_pred.to_csv('data/results_folds.csv', header=True)

    # print('results', results.max(), results.min())

    test_pred = clean_results(test_pred)

    mae = abs(test_pred['pred'] - test_pred['ground']).mean()
    mse = mean_squared_error(test_pred['pred'], test_pred['ground'])
    rmse = mean_squared_error(test_pred['pred'], test_pred['ground'], squared=False)
    mae = mean_absolute_error(test_pred['pred'], test_pred['ground'])

    # print(test_measures)
    # print("mae", mae)
    # print("mse", mse)

    test_measures.append(predicts)
    test_measures.append(emb)
    test_measures.append(mse)
    test_measures.append(rmse)
    test_measures.append(mae)
    test_measures.append(test_pred)

    # print("results")
    # print(results_df)
    # sys.exit()

    return test_measures


def save_results_folds(rmses, mses, maes):
    mses = np.array(mses)
    rmses = np.array(rmses)
    maes = np.array(maes)

    rmse = np.mean(rmses).round(20)
    mse = np.mean(mses).round(20)
    mae = np.mean(maes).round(20)

    # columns = index,assignment,fold,layer,features,rmse_GCN,mse_GCN
    df = pd.read_csv('results/results_self_loop.csv')

    # df = df.drop(columns=['index'])

    self_loop_int = 0
    if self_loop:
        self_loop_int = 1

    df.loc[len(df.index)] = [assignment, fold, self_loop_int, layer, node_feature, rmse, mse, mae]

    df = df.round(20)

    # df.to_csv('results/results_self_loop.csv', index=False)
    print("saved")

    return df


def save_results_assignments(fold_results):
    _results = pd.read_csv('results/results.csv', index_col='index')

    for label, group in fold_results.groupby(['features', 'layer', 'assignment']):
        row = []

        row.append(label[0])
        row.append(label[1])
        row.append(label[2])
        row.append(group['rmse_GCN'].mean().round(20))
        row.append(group['mse_GCN'].mean().round(20))
        row.append(group['mae_GCN'].mean().round(20))

        _results.loc[len(_results.index)] = row

    _results = _results.round(20)
    _results.to_csv('results/results.csv')

    return _results


def get_assignment_median(G_df=None, torch_id=None):
    return G_df[G_df['target_id'] == torch_id]['weight'].median()


def get_assignment_average(G_df=None, torch_id=None):
    return round(G_df[G_df['target_id'] == torch_id]['weight'].mean(), 20)


def compare_models(results, G_df):
    results['median_grade'] = results.apply(lambda row: get_assignment_median(G_df=G_df, torch_id=row['id']),
                                            axis=1)
    results['average_grade'] = results.apply(lambda row: get_assignment_average(G_df=G_df, torch_id=row['id']),
                                             axis=1)

    results = results.round(20)
    results.loc[results['pred'] < 0, 'pred'] = 0
    results.loc[results['pred'] > 1, 'pred'] = 1

    rmse_gcn = mean_squared_error(results['ground'], results['pred'], squared=False)
    rmse_median = mean_squared_error(results['ground'], results['median_grade'], squared=False)
    rmse_average = mean_squared_error(results['ground'], results['average_grade'], squared=False)

    mse_gcn = mean_squared_error(results['ground'], results['pred'])
    mse_median = mean_squared_error(results['ground'], results['median_grade'])
    mse_average = mean_squared_error(results['ground'], results['average_grade'])

    mae_gcn = mean_absolute_error(results['ground'], results['pred'])
    mae_median = mean_absolute_error(results['ground'], results['median_grade'])
    mae_average = mean_absolute_error(results['ground'], results['average_grade'])

    row = {
        "rmse_gcn": rmse_gcn,
        "rmse_average": rmse_average,
        "rmse_median": rmse_median,
        "mse_gcn": mse_gcn,
        "mse_median": mse_median,
        "mse_average": mse_average,
        "mae_gcn": mae_gcn,
        "mae_median": mae_median,
        "mae_average": mae_average
    }

    df_compare_models = pd.DataFrame([row])

    # df_compare_models = df_compare_models.round(6)

    print("---------------------")
    print(df_compare_models)
    print("---------------------")
    print()
    print()
    print()
    # sys.exit()


def clean_results(results):
    greater_one = results[results['pred'] > 1].index
    smaller_one = results[results['pred'] < 0].index
    results.loc[greater_one, 'pred'] = 1
    results.loc[smaller_one, 'pred'] = 0

    # results['pred'] = results['pred'].abs()

    results = results.round(20)

    return results


def save_predicted_grades_to_experiments(results, experiment_id=None, fold=None):
    simulated_df = gcn_data_loader_no_val.get_simulated_df(experiment_id=experiment_id)

    column_name = 'gcn_fold' + str(fold)
    for index, row in results.iterrows():
        student_index = row['id'] - simulated_df.shape[0]
        simulated_df.loc[student_index, column_name] = row['pred'].round(20)

    simulated_df = simulated_df.round(20)

    simulated_df.to_csv(ROOT_DIR + 'data/simulated_studies/study6/{}.csv'.format(experiment_id), index=False)

    return 1


def run_gcn(experiment_id=None, fold=None, params=None, best_params=False):

    mses = dict()
    rmses = dict()
    maes = dict()

    data, G_df = prepare_data_for_gcn(experiment_id=experiment_id, fold=fold)

    model, data, optimizer = define_model(data, layer=params['number_of_layer'], params=params)

    best_test_acc = test_acc = 100
    best_epoch = 1
    for epoch in range(1, params["epochs"]):
        train(model=model, optimizer=optimizer, data=data)
        train_acc, tmp_test_acc, predicts, embeddings, mse, rmse, mae, test_preds = test(
            model=model, data=data)
        test_acc = tmp_test_acc
        log = 'Epoch: {:03d}, Train: {:.4f}, Test: {:.4f}'
        if homophily_info:
            print('cf:', homophily_info['control_friendship'], 'run:', homophily_info['run'], 'fold:', fold, 'epoch:', epoch, 'train mse:', train_acc, 'test mse', test_acc)
        else:
            print('fold:', fold, 'epoch:', epoch, 'train mse:', train_acc, 'test mse', test_acc)

        if test_acc < best_test_acc and epoch > 75:
            best_test_acc = test_acc
            best_test_mse = test_acc
            best_test_rmse = rmse
            best_epoch = epoch


        results_df = test_preds
        last_test_acc = test_acc
        last_test_rmse = rmse

    print("best_epoch:", best_epoch,
          "best_test_mse:", best_test_mse,
          "best_test_rmse:", best_test_rmse,
          "last_test_rmse:", rmse)

    # sys.exit()

    if best_params:
        save_predicted_grades_to_experiments(results_df, experiment_id, fold)

    return best_test_rmse, last_test_rmse


def run(_experiment_id=None, _homophily_info=None, params=None):

    global experiment_id
    experiment_id = _experiment_id

    global homophily_info
    homophily_info = _homophily_info


    manual_params = {"epochs": 800, "dimension": 64, "learning_rate": 0.02, "number_of_layer": 3}

    if params:
        manual_params = params

    all_test_rmses = []
    for fold in folds:
        temp_val_loss, temp_test_rmse = run_gcn(experiment_id=experiment_id, fold=fold,
                                                params=manual_params, best_params=True)
        all_test_rmses.append(temp_test_rmse)


    print("Best test rmse for friendship({}) and ownership({}):".format(int(friendship), int(ownership)))
    print(np.mean(all_test_rmses))

    return np.mean(all_test_rmses)


if __name__ == '__main__':
    run()




