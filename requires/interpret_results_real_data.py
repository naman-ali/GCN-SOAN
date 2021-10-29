import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
from requires import helper
from requires.config import ROOT_DIR, MODELS, study


REAL_DATA_PROCESSED_DIR = ROOT_DIR + 'data/real_data_processed/'

if study == 'real_data_peer_evaluation':
    rmse_models_file = 'rmse_models_peer.csv'

if study == 'real_data_peer_and_self_evaluation':
    rmse_models_file = 'rmse_models_peer_self.csv'


def load_folds(assign=None, fold=None, which='test'):

    fold_df = pd.read_csv(REAL_DATA_PROCESSED_DIR + 'folds/assign{}-fold{}-{}.csv'.format(assign, fold, which))

    return fold_df


def get_task_id(row):
    return row['assignid'].split('-')[0]


def get_group_id(row):
    return row['assignid'].split('-')[1]


def load_predicted_grades():
    return pd.read_csv(REAL_DATA_PROCESSED_DIR + 'gg-models-results.csv')


def interpret_avg():

    results = load_predicted_grades()
    results['assignid'] = results['task_id'].astype(str) + "-" + results['group_id'].astype(str)

    rmses = dict()

    assignments = [1, 2, 3, 4]
    folds = [1, 2, 3, 4, 5]
    for assignment in assignments:
        rmses[assignment] = dict()
        for fold in folds:
            folds_df = load_folds(assign=assignment, fold=fold)

            results_fold = results[results['assignid'].isin(folds_df['assignid'].values)]

            rmse = mean_squared_error(results_fold['TA_grade'], results_fold['avg_grade'], squared=False)

            rmses[assignment][fold] = round(rmse, 5)


    return update_rmse_models_file(rmses, model='avg')


def interpret_fold_models(model=None):

    results = pd.read_csv(ROOT_DIR+'/results/{}.csv'.format(model))
    results['assignid'] = results['taskid'].astype(str) + "-" + results['group_id'].astype(str)

    # rmses = dict()
    # for label, group in results.groupby(['assignid', 'fold']):
    #
    #     rmses[label[0]] = dict()
    #
    #     mse = mean_squared_error(group['ta_grade'], group['grade'], squared=True)
    #     rmse = mean_squared_error(group['ta_grade'], group['grade'], squared=False)
    #     mae = mean_absolute_error(group['ta_grade'], group['grade'])
    #
    #     rmses[label[0]][label[1]] = rmse
    #
    # return update_rmse_models_file(rmses, model='RankwithTA')

    rmses = dict()
    assignments = [1, 2, 3, 4]
    folds = [1, 2, 3, 4]
    for assignment in assignments:
        rmses[assignment] = dict()
        for fold in folds:
            folds_df = load_folds(assign=assignment, fold=fold)

            results_fold = results[results['assignid'].isin(folds_df['assignid'].values)]

            rmse = mean_squared_error(results_fold['ta_grade'], results_fold['grade'], squared=False)

            rmses[assignment][fold] = round(rmse, 5)


    return update_rmse_models_file(rmses, model=model)


def interpret_model(model=None):

    results = pd.read_csv(REAL_DATA_PROCESSED_DIR + 'gg-models-results.csv')

    results = results.dropna()

    results['assignid'] = results['task_id'].astype(str) + "-" + results['group_id'].astype(str)

    rmses = dict()

    assignments = [1, 2, 3, 4]
    folds = [1, 2, 3, 4]
    for assignment in assignments:
        rmses[assignment] = dict()
        for fold in folds:
            folds_df = load_folds(assign=assignment, fold=fold)

            results_fold = results[results['assignid'].isin(folds_df['assignid'].values)]

            rmse = mean_squared_error(results_fold['TA_grade'], results_fold[model+'_grade'], squared=False)

            rmses[assignment][fold] = round(rmse, 5)

    return update_rmse_models_file(rmses, model=model)



def update_rmse_models_file(rmses, model=None):

    rmse_models = load_rmse_models()

    for assignment, rmse_folds in rmses.items():
        mean_rmse = round(np.mean(list(rmse_folds.values())), 5)
        rmse_models.loc[rmse_models['assignment'] == assignment, model] = mean_rmse

    rmse_models = rmse_models.round(5)

    rmse_models.to_csv(REAL_DATA_PROCESSED_DIR + rmse_models_file, index=False)

    return rmse_models


def create_rmse_models_file():

    if not helper.file_exist(REAL_DATA_PROCESSED_DIR + rmse_models_file):
        df = pd.DataFrame(columns=['assignment', 'Average', 'Median', 'PeerRank', 'TunedModel', 'RankwithTA', 'GCNSOAN', 'Vancouver'])
        for assignment in [1, 2, 3, 4]:
            df.loc[df.shape[0], 'assignment'] = assignment

        df.to_csv(REAL_DATA_PROCESSED_DIR + rmse_models_file, index=False)


def load_rmse_models():

    df = pd.read_csv(REAL_DATA_PROCESSED_DIR + rmse_models_file)

    return df


def run(gcn_rmses=None):

    create_rmse_models_file()

    for model in MODELS:

        if model != 'GCN':

            if model not in ['RankwithTA']:
                interpret_model(model=model)

            if model in ['RankwithTA']:
                interpret_fold_rmses(model=model)


    rmse_models = load_rmse_models()

    # add gcn result
    if gcn_rmses:
        for assignment, rmse in gcn_rmses.items():
            print(assignment, rmse)
            rmse_models.loc[rmse_models['assignment'] == int(assignment), 'GCNSOAN'] = rmse

        rmse_models['best'] = rmse_models.idxmin(axis=1)
        rmse_models.to_csv(REAL_DATA_PROCESSED_DIR + rmse_models_file, index=False)

    return rmse_models.round(5)



def taskid_to_assignment(taskid=None):

    x = [int(a) for a in str(int(taskid))]
    return x[0]


def interpret_fold_rmses(model='RankwithTA'):

    if model == 'RankwithTA':
        results = pd.read_csv(REAL_DATA_PROCESSED_DIR + '{}.csv'.format(model))

    results['assignment'] = results.apply(lambda row: taskid_to_assignment(taskid=row['taskid']), axis=1)

    rmses = {
        1: dict(),
        2: dict(),
        3: dict(),
        4: dict()
    }

    for label, fold in results.groupby(['assignment', 'fold']):

        rmse = mean_squared_error(fold['grade'], fold['ta_grade'], squared=False)
        rmses[label[0]][label[1]] = round(rmse, 10)

    return update_rmse_models_file(rmses, model=model)


if __name__ == '__main__':

    run()
    # interpret_fold_rmses(model='SemiAverage')

