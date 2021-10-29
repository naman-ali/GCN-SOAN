import pandas as pd
import matplotlib
from requires.config import *

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

from sklearn.metrics import mean_squared_error


def experiment_rmse(folds=None, experiment_id=None):

    experiment_file = STUDY_DIR + str(experiment_id) + '.csv'

    df = pd.read_csv(experiment_file)

    rmses = dict()
    for model in MODELS:
        rmses[model] = []


    for fold in folds:

        gcn_column_name = 'gcn_fold'+str(fold)
        RankwithTA_column_name = 'RankwithTA_fold'+str(fold)

        df_fold = df[df[gcn_column_name].notnull()]

        rmses['Average'].append(mean_squared_error(df_fold['Average'], df_fold['actual_grade'], squared=False))
        rmses['Median'].append(mean_squared_error(df_fold['Median'], df_fold['actual_grade'], squared=False))
        rmses['PeerRank'].append(mean_squared_error(df_fold['PeerRank'], df_fold['actual_grade'], squared=False))
        rmses['TunedModel'].append(mean_squared_error(df_fold['TunedModel'], df_fold['actual_grade'], squared=False))
        rmses['Vancouver'].append(mean_squared_error(df_fold['Vancouver'], df_fold['actual_grade'], squared=False))
        rmses['RankwithTA'].append(mean_squared_error(df_fold[RankwithTA_column_name], df_fold['actual_grade'], squared=False))
        rmses['GCN'].append(mean_squared_error(df_fold[gcn_column_name], df_fold['actual_grade'], squared=False))

    mean_rmses = dict()
    for model in rmses.keys():
        mean_rmses[model] = np.round(np.mean(rmses[model]), 5)


    exp_inf = pd.read_csv(EXPERIMENT_INFO_FILE)


    experiment_index = exp_inf[exp_inf['experiment_id'] == experiment_id].index


    exp_inf.loc[experiment_index, 'average_rmse'] = mean_rmses['Average']
    exp_inf.loc[experiment_index, 'median_rmse'] = mean_rmses['Median']
    exp_inf.loc[experiment_index, 'peerrank_rmse'] = mean_rmses['PeerRank']
    exp_inf.loc[experiment_index, 'TunedModel_rmse'] = mean_rmses['TunedModel']
    exp_inf.loc[experiment_index, 'Vancouver_rmse'] = mean_rmses['Vancouver']
    exp_inf.loc[experiment_index, 'RankwithTA_rmse'] = mean_rmses['RankwithTA']
    exp_inf.loc[experiment_index, 'gcn_rmse'] = mean_rmses['GCN']

    exp_inf = exp_inf.round(5)

    exp_inf.to_csv(EXPERIMENT_INFO_FILE, index=False)


def get_experiments_ids(which='not_calculated'):
    experiments_info = pd.read_csv(EXPERIMENT_INFO_FILE)

    if which == 'not_calculated':
        experiments_info = experiments_info[experiments_info['gcn_rmse'].isnull()]
    else:
        experiments_info = experiments_info[experiments_info['study'] == which]

    return experiments_info.experiment_id.values


def load_experiment_infos():
    exp_inf = pd.read_csv(EXPERIMENT_INFO_FILE)
    return exp_inf


def run(study=None, experiments=None, x_axis='alpha', ):


    experiment_ids = get_experiments_ids(which=study)

    if experiments:
        experiment_ids = experiments

    for experiment_id in experiment_ids:
        experiment_rmse(folds=range(1, 4), experiment_id=experiment_id)
