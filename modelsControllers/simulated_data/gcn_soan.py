import torch
import torch.nn.functional as F
from modelsControllers.simulated_data.GCNSOANConv import GCNSOANConv
# from torch_geometric.nn import GCNConv, ChebConv  # noqa
from torch_geometric.data import Data
from requires.gcn_data_loader_simulated import get_simulated_data, get_simulated_df
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd

from requires.config import *

folds = [1, 2, 3, 4]
ownership = False
friendship = False
layer = 2
epochs = 800
runs = 3
dimension = 64
learning_rate = 0.02
test_size = 0.9


if study == 'Study 6: homophily' or study == 'Study 5: erdos':
    ownership = True
    friendship = True
    layer = 3

fold_results = []



def prepare_data_for_gcn(experiment_id=None, fold=None):
    features, labels, G_df, folds = get_simulated_data(experiment_id=experiment_id,
                                                       ownership=ownership,
                                                       friendship=friendship,
                                                       test_size=test_size
                                                       )

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

    # print(test_idx.shape, train_idx.shape)
    # sys.exit()

    return data, G_df


class Net(torch.nn.Module):

    def __init__(self, data):
        super(Net, self).__init__()

        self.data = data

        num_features = data.x.size(1)
        num_classes = torch.unique(data.y).size(0)

        if layer == 2:
            self.conv1 = GCNSOANConv(num_features, dimension, cached=False, normalize=True)
            self.conv2 = GCNSOANConv(dimension, 1, cached=False, normalize=True)

        if layer == 3:
            self.conv1 = GCNSOANConv(num_features, dimension, cached=False, normalize=True)
            self.conv2 = GCNSOANConv(dimension, dimension, cached=False, normalize=True)
            self.conv3 = GCNSOANConv(dimension, 1, cached=False, normalize=True)

    def forward(self):
        x, edge_index, edge_weight = self.data.x, self.data.edge_index, self.data.edge_attr

        if layer == 2:
            x = F.elu(self.conv1(x, edge_index, edge_weight))
            x = F.dropout(x, training=self.training, p=0.0)
            x = torch.sigmoid(self.conv2(x, edge_index, edge_weight))

        if layer == 3:
            x = F.elu(self.conv1(x, edge_index, edge_weight))
            x = F.dropout(x, training=self.training, p=0.0)
            x = F.elu(self.conv2(x, edge_index, edge_weight))
            emb = x
            x = torch.sigmoid(self.conv3(x, edge_index, edge_weight))

        return x


def define_model(data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, data = Net(data).to(device), data.to(device)

    if layer == 2:
        optimizer = torch.optim.Adam([
            dict(params=model.conv1.parameters(), weight_decay=0),
            dict(params=model.conv2.parameters(), weight_decay=0)
        ], lr=learning_rate)  # Only perform weight-decay on first convolution.

    if layer == 3:
        optimizer = torch.optim.Adam([
            dict(params=model.conv1.parameters(), weight_decay=0),
            dict(params=model.conv2.parameters(), weight_decay=0),
            dict(params=model.conv3.parameters(), weight_decay=0)
        ], lr=learning_rate)  # Only perform weight-decay on first convolution.

    return model, data, optimizer


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
    test_measures = []

    test_pred = pd.DataFrame()
    node_tarcnets = np.ones((1, 1))

    for _, mask in data('train_mask', 'test_mask'):

        pred = torch.reshape(predicts[mask], (-1,))

        if _ == 'test_mask':
            test_pred['pred'] = pred.numpy()
            test_pred['ground'] = data.y[mask].numpy()
            test_pred['id'] = data.test_idx

        loss = F.mse_loss(pred, data.y[mask])

        test_measures.append(round(loss.item(), 20))

    test_pred = clean_results(test_pred)

    mae = abs(test_pred['pred'] - test_pred['ground']).mean()
    mse = mean_squared_error(test_pred['pred'], test_pred['ground'])
    rmse = mean_squared_error(test_pred['pred'], test_pred['ground'], squared=False)
    mae = mean_absolute_error(test_pred['pred'], test_pred['ground'])

    test_measures.append(predicts)
    test_measures.append(mse)
    test_measures.append(rmse)
    test_measures.append(mae)
    test_measures.append(test_pred)

    return test_measures


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
    results['median_grade'] = results.apply(lambda row: get_assignment_median(G_df=G_df, torch_id=row['id']), axis=1)
    results['average_grade'] = results.apply(lambda row: get_assignment_average(G_df=G_df, torch_id=row['id']), axis=1)
    results = results.round(20)
    rmse_gcn = mean_squared_error(results['ground'], results['pred'], squared=False)
    rmse_median = mean_squared_error(results['ground'], results['median_grade'], squared=False)
    rmse_average = mean_squared_error(results['ground'], results['average_grade'], squared=False)
    row = {
        "rmse_gcn": rmse_gcn,
        "rmse_average": rmse_average,
        "rmse_median": rmse_median,
    }
    df_compare_models = pd.DataFrame([row])
    df_compare_models = df_compare_models.round(20)

    print("\n\n\n---------------------")
    print(df_compare_models)
    print("---------------------\n\n\n")


def clean_results(results):
    greater_one = results[results['pred'] > 1].index
    smaller_one = results[results['pred'] < 0].index
    results.loc[greater_one, 'pred'] = 1
    results.loc[smaller_one, 'pred'] = 0

    results = results.round(20)

    return results


def save_predicted_grades_to_experiments(results, experiment_id=None, fold=None):
    simulated_df = get_simulated_df(experiment_id=experiment_id)

    column_name = 'gcn_fold' + str(fold)
    for index, row in results.iterrows():
        student_index = row['id'] - simulated_df.shape[0]
        simulated_df.loc[student_index, column_name] = row['pred'].round(20)

    simulated_df = simulated_df.round(20)

    experiment_file = STUDY_DIR + str(experiment_id) + '.csv'

    simulated_df.to_csv(experiment_file, index=False)

    print("save_predicted_grades_to_experiments")
    print(simulated_df.head(2))
    # sys.exit()

    return 1


def set_params(params=None):
    global layer
    global ownership
    global friendship
    global epochs
    global runs
    global dimension
    global learning_rate
    global test_size

    layer = params['number_of_layer']
    ownership = params['ownership']
    friendship = params['friendship']
    epochs = params['epochs']
    runs = params['runs']
    dimension = params['dimension']
    learning_rate = params['learning_rate']
    test_size = params['test_size']

    return True


predicts = None

step = 'generate_scratch'


# step = 'average_folds'


def run(experiment_id=None, params=None):
    if params:
        set_params(params)

    if step == 'generate_scratch':

        rmses = []
        for fold in folds:

            print("fold", fold)


            data, G_df = prepare_data_for_gcn(experiment_id=experiment_id, fold=fold)

            for run in range(0, runs):

                model, data, optimizer = define_model(data)

                for epoch in range(1, epochs):
                    train(model=model, optimizer=optimizer, data=data)
                    train_acc, test_acc, predicts, mse, rmse, mae, test_preds = test(model=model, data=data)
                    print(epoch, 'train mse:', train_acc, 'test mse', test_acc)

                    if epoch == epochs - 1:
                        rmses.append(round(rmse, 20))
                        results_df = test_preds

                compare_models(results_df, G_df)

                save_predicted_grades_to_experiments(results_df, experiment_id, fold)

        return np.mean(rmses)

    if step == 'average_folds':
        fold_results = pd.read_csv('results/results_folds.csv', index_col='index')
        save_results_assignments(fold_results)


if __name__ == '__main__':
    run(experiments=range(51, 52))
