import matplotlib.pyplot as plt
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

    rmses = {
        'Average': [],
        'Median': [],
        'PeerRank': [],
        'TunedModel': [],
        'Vancouver': [],
        'RankwithTA': [],
        'GCN': []
    }

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


    exp_inf.loc[experiment_index, 'Average_rmse'] = mean_rmses['Average']
    exp_inf.loc[experiment_index, 'Median_rmse'] = mean_rmses['Median']
    exp_inf.loc[experiment_index, 'PeerRank_rmse'] = mean_rmses['PeerRank']
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


def plot_varies_x(x_axis=None, study=None, experiments=None, TunedModel_plot=True):
    plot_label = ''
    if x_axis == 'alpha':
        plot_label = r'Bias parameter $\alpha$'
    if x_axis == 'mean2':
        plot_label = r'Ground truth mean $\mu$'
    if x_axis == 'number_of_graders':
        plot_label = r'Number of graders per item $k$'
    if x_axis == 'beta':
        plot_label = r'Reliability parameter $\beta$'
    # if x_axis == 'beta':
    #     plot_label = 'Varying graders errors(standard deviation of peer grades distribution)'
    if x_axis == 'control_friendship':
        plot_label = 'Connection probability p'


    exp_info = load_experiment_infos()

    exp_info = exp_info[exp_info['study'] == study]
    # remember to comment this line
    if experiments:
        exp_info = exp_info[exp_info['experiment_id'].isin(experiments)]

    exp_info = exp_info.sort_values(by=x_axis)

    columns = exp_info.columns.tolist()
    columns.remove(x_axis)
    columns.remove('experiment_id')
    columns.remove('Average_rmse')
    columns.remove('Median_rmse')
    columns.remove('PeerRank_rmse')
    if 'TunedModel_rmse' in columns:
        columns.remove('TunedModel_rmse')
    columns.remove('Vancouver_rmse')
    columns.remove('RankwithTA_rmse')
    columns.remove('gcn_rmse')

    print(exp_info.shape)

    runs = dict()
    runs['Average_rmse'] = dict()
    runs['Median_rmse'] = dict()
    runs['PeerRank_rmse'] = dict()
    if TunedModel_plot:
        runs['TunedModel_rmse'] = dict()
    runs['Vancouver_rmse'] = dict()
    runs['RankwithTA_rmse'] = dict()
    runs['gcn_rmse'] = dict()

    for label, group in exp_info.groupby([x_axis]):
        runs['Average_rmse'][label] = list()
        runs['Median_rmse'][label] = list()
        runs['PeerRank_rmse'][label] = list()
        if TunedModel_plot:
            runs['TunedModel_rmse'][label] = list()
        runs['Vancouver_rmse'][label] = list()
        runs['RankwithTA_rmse'][label] = list()
        runs['gcn_rmse'][label] = list()

    for label, group in exp_info.groupby([x_axis]):
        runs['Average_rmse'][label].append(group['Average_rmse'].mean().round(4))
        runs['Median_rmse'][label].append(group['Median_rmse'].mean().round(4))
        runs['PeerRank_rmse'][label].append(group['PeerRank_rmse'].mean().round(4))
        if TunedModel_plot:
            runs['TunedModel_rmse'][label].append(group['TunedModel_rmse'].mean().round(4))
        runs['Vancouver_rmse'][label].append(group['Vancouver_rmse'].mean().round(4))
        runs['RankwithTA_rmse'][label].append(group['RankwithTA_rmse'].mean().round(4))
        runs['gcn_rmse'][label].append(group['gcn_rmse'].mean().round(4))

    fig, ax = plt.subplots(figsize=(7, 6))

    x_values = exp_info[x_axis].unique()

    ax.plot(x_values, runs['Average_rmse'].values(), label="Average", color='#ff9933', linestyle='solid', marker='^',
            markersize='7', alpha=1.0)
    ax.plot(x_values, runs['Median_rmse'].values(), label="Median", color='#8f8f8f', linestyle='solid', marker='v',
            markersize='7', alpha=1.0)
    ax.plot(x_values, runs['PeerRank_rmse'].values(), label="PeerRank", color='#00b8e6', linestyle='solid', marker='*',
            markersize='9', alpha=1.0)
    if TunedModel_plot:
        ax.plot(x_values, runs['TunedModel_rmse'].values(), label="TunedModel", color='black', linestyle='solid', marker='D',
            markersize='6', alpha=1.0)
    ax.plot(x_values, runs['Vancouver_rmse'].values(), label="Vancouver", color='#f44336', linestyle='solid',
            marker='D', markersize='6', alpha=1.0)
    ax.plot(x_values, runs['RankwithTA_rmse'].values(), label="RankwithTA", color='#cc33ff', linestyle='solid',
                marker='P', markersize='6', alpha=1.0)

    ax.plot(x_values, runs['gcn_rmse'].values(), label="GCN-SOAN", color='#00cc99', linestyle='solid', marker='o',
            markersize='7', alpha=1.0)

    ax.legend(fontsize=15, ncol=2)

    plt.yticks(fontsize=15)
    plt.xticks(np.sort(exp_info[x_axis].unique()), rotation='horizontal', fontsize=13)
    plt.xlabel(plot_label, fontsize=17)
    plt.ylabel("RMSE", fontsize=17)
    # ax.set_ylim((0.045, 0.17))

    plt.savefig(STUDY_DIR + 'study' + str(STUDY_NUMBER) + '.pdf', dpi=300)

    plt.show()
    plt.close()
    # sys.exit()


def run(study=None, experiments=None, x_axis='alpha'):

    experiment_ids = get_experiments_ids(which=study)

    if experiments:
        experiment_ids = experiments

    for experiment_id in experiment_ids:
        experiment_rmse(folds=range(1, 4), experiment_id=experiment_id)

    TunedModel_plot = True
    if 'TunedModel' not in MODELS:
        TunedModel_plot = False

    plot_varies_x(x_axis=x_axis, study=study, experiments=experiment_ids, TunedModel_plot=TunedModel_plot)


