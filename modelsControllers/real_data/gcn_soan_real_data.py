import torch
import torch.nn.functional as F
# from torch_geometric.nn import GCNConv, ChebConv  # noqa
from modelsControllers.simulated_data.GCNSOANConv import GCNSOANConv
from torch_geometric.data import Data
import numpy as np
from requires.gcn_data_loader_real import get_real_data
from sklearn.metrics import mean_squared_error
import pandas as pd
from requires.config import study, fast_gcn

assignments = {
    '1': ['11', '12', '13'],
    '2': ['21', '22', '23', '24'],
    '3': ['31', '32', '33', '34', '35'],
    '4': ['41', '42', '43']
}
folds = ['1', '2', '3', '4']
peer_evaluation = True

if study == 'real_data_peer_evaluation':
    self_evaluation = False

if study == 'real_data_peer_and_self_evaluation':
    self_evaluation = True

layers = 2
node_features = 'one'
ownership = False
sigmoid = True
test_size = 0.8
runs = 3
epochs = 800
dimension = 64
learning_rate = 0.02

if fast_gcn:
    epochs = 10
    runs = 1


fold_results = []


def check_duplicate_edges(_G):
    dup = _G[_G.duplicated(subset=['source', 'target'])]


def prepare_real_data_for_gcn():
    features, labels, G_df, folds = get_real_data(taskid=taskid,
                                                  assignment=assignment,
                                                  ownership=ownership,
                                                  self_evaluation=self_evaluation,
                                                  peer_evaluation=peer_evaluation,
                                                  test_size=test_size)

    check_duplicate_edges(G_df)

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

    data.test_mask = torch.tensor(np.full(labels.shape, False), dtype=torch.bool)
    data.test_mask[test_idx] = True

    assignment_idx = (data.y != -1).nonzero(as_tuple=False).view(-1)
    data.assign_mask = torch.tensor(np.full(labels.shape, False), dtype=torch.bool)
    data.assign_mask[assignment_idx] = True

    data.test_idx = test_idx

    return data, G_df


class Net(torch.nn.Module):
    def __init__(self, data):
        super(Net, self).__init__()

        self.data = data

        num_features = data.x.size(1)
        num_classes = torch.unique(data.y).size(0)

        self.conv1 = GCNSOANConv(num_features, dimension, cached=False, normalize=True)
        self.conv2 = GCNSOANConv(dimension, 1, cached=False, normalize=True)


    def forward(self):

        x, edge_index, edge_weight = self.data.x, self.data.edge_index, self.data.edge_attr

        x = F.elu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training, p=0.0)
        x = torch.sigmoid(self.conv2(x, edge_index, edge_weight))

        return x


def define_model(data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, data = Net(data).to(device), data.to(device)

    optimizer = torch.optim.Adam([
            dict(params=model.conv1.parameters(), weight_decay=0),
            dict(params=model.conv2.parameters(), weight_decay=0)
        ], lr=learning_rate)  # Only perform weight-decay on first convolution.

    return model, data, optimizer


def clean_results(results):
    greater_one = results[results['pred'] > 1].index
    smaller_one = results[results['pred'] < 0].index
    results.loc[greater_one, 'pred'] = 1
    results.loc[smaller_one, 'pred'] = 0

    results = results.round(20)

    return results


def train(model=None, optimizer=None, data=None):
    model.train()
    optimizer.zero_grad()
    model_ = model()
    model_res = torch.reshape(model_[data.train_mask], (-1,))
    F.mse_loss(model_res, data.y[data.train_mask]).backward()
    optimizer.step()


@torch.no_grad()
def test(model=None, data=None):
    model.eval()
    predicts = model()
    test_results = []

    test_pred = pd.DataFrame()
    node_tarcnets = np.ones((1, 1))
    for _, mask in data('train_mask', 'test_mask'):
        pred = torch.reshape(predicts[mask], (-1,))

        if _ == 'test_mask':
            test_pred['pred'] = pred.numpy()
            test_pred['ground'] = data.y[mask].numpy()
            test_pred['id'] = data.test_idx

        loss = F.mse_loss(pred, data.y[mask])

        test_results.append(loss)

    test_pred = clean_results(test_pred)
    rmse = mean_squared_error(test_pred['pred'], test_pred['ground'], squared=False)

    test_results.append(rmse)
    test_results.append(test_pred)

    return test_results


def get_assignment_median(G_df=None, torch_id=None):
    return G_df[G_df['target_id'] == torch_id]['weight'].median()


def get_assignment_average(G_df=None, torch_id=None):

    G_df = G_df[G_df['edge_type'] == 'grade']

    return round(G_df[G_df['target_id'] == torch_id]['weight'].mean(), 2)


def compare_models(results, G_df):

    results['median_grade'] = results.apply(lambda row: get_assignment_median(G_df=G_df, torch_id=row['id']), axis=1)
    results['average_grade'] = results.apply(lambda row: get_assignment_average(G_df=G_df, torch_id=row['id']), axis=1)

    rmse_gcn = mean_squared_error(results['ground'], results['pred'], squared=False)
    rmse_median = mean_squared_error(results['ground'], results['median_grade'], squared=False)
    rmse_average = mean_squared_error(results['ground'], results['average_grade'], squared=False)

    row = {
        "rmse_gcn": rmse_gcn,
        "rmse_median": rmse_median,
        "rmse_average": rmse_average
    }

    df_compare_models = pd.DataFrame([row])

    df_compare_models = df_compare_models.round(6)

    print(df_compare_models)

    return df_compare_models


def run():
    global layer
    global taskid
    global assignment
    global fold

    assignments_rmses = dict()
    for assignment, taskid in assignments.items():

        assignment = str(assignment)

        folds_rmses = []
        for fold in folds:

            data, G_df = prepare_real_data_for_gcn()

            runs_rmses = []
            for run in range(0, runs):

                model, data, optimizer = define_model(data)

                for epoch in range(1, epochs):
                    train(model=model, optimizer=optimizer, data=data)
                    train_acc, test_acc, rmse, results_df = test(model=model, data=data)

                    print(epoch, 'train mse:', train_acc.item(), 'test mse', test_acc.item())

                    if epoch == epochs - 1:
                        runs_rmses.append(round(rmse, 5))

                results_df = results_df.round(20)

                compare_models_df = compare_models(results_df, G_df)

            folds_rmses.append(np.mean(runs_rmses))

            print("folds_rmses:", folds_rmses)
            # fold_results = save_results_folds(rmses, compare_models_df)

        assignments_rmses[assignment] = np.mean(folds_rmses)

    return assignments_rmses


if __name__ == '__main__':
    run()
