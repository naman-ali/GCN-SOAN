import pandas as pd
# from models.gcn_pg_optuna_no_val import run as gcn_run
from modelsControllers.simulated_data.gcn_soan import run as gcn_run
from sklearn.metrics import mean_squared_error
import numpy as np
import os
import matplotlib
from modelsControllers.simulated_data.Average import Average
from modelsControllers.simulated_data.Median import Median
from modelsControllers.simulated_data.PeerRank import PeerRank
from modelsControllers.simulated_data.TunedModel import TunedModel
from modelsControllers.simulated_data.RankwithTA import run as RankwithTA_run
from modelsControllers.simulated_data.vancouver import run as Vancouver_run
from requires.config import ROOT_DIR, STUDY_NUMBER, STUDY_DIR, MODELS, EXPERIMENT_INFO_FILE
import sys

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


experiment_ids = [1]
experiment_id = experiment_ids[0]

control_friendships = [0.0001, 0.001, 0.01, 0.1]

param = {
    "epochs": 800,
    "dimension": 64,
    "learning_rate": 0.02,
    "number_of_layer": 3,
    "friendship": True,
    "ownership": True,
    "runs": 3,
    "test_size": 0.9
}


def load_grades():
    df = pd.read_csv(STUDY_DIR + str(experiment_id) + '.csv')
    df['friends'] = ''
    return df


def update_friendships(_df_grades):
    return _df_grades.to_csv(STUDY_DIR + str(experiment_id) + '.csv', index=False)


def generate_friendships(_df_grades, control_friendship=None):

    for index, row in _df_grades.iterrows():
        grade = row['actual_grade']
        friends = _df_grades[(_df_grades['actual_grade'] <= grade + control_friendship) &
                             (_df_grades['actual_grade'] >= grade - control_friendship)]

        if not friends.empty:
            friends_list = list(friends.index)
            friends_list.remove(index)
            friends_str = ','.join([str(i) for i in friends_list])

        _df_grades.loc[index, 'friends'] = friends_str

    return _df_grades


def load_experiment_infos():
    exp_inf = pd.read_csv(EXPERIMENT_INFO_FILE)
    return exp_inf


def run():

    for _experiment_id in experiment_ids:

        global experiment_id
        experiment_id = _experiment_id

        # Baselines
        df = load_grades()
        Average(_experiment_id, df)
        Median(_experiment_id, df)
        PeerRank(_experiment_id, df)
        TunedModel(_experiment_id, df)
        RankwithTA_run([_experiment_id])
        Vancouver_run(_experiment_id)

        from requires.interpret_results import experiment_rmse, load_experiment_infos
        experiment_rmse(folds=range(1, 4), experiment_id=experiment_id)

        # GCN-SOAN
        df_grades = load_grades()
        gcn_rmses = dict()

        for control_friendship in control_friendships:

            gcn_rmses[str(control_friendship)] = []

            for run in range(1, param['runs']+1):

                # generate friendships
                df_grades = generate_friendships(df_grades, control_friendship=control_friendship)

                # update experiment_id.csv file with new friendships
                update_friendships(df_grades)

                # run GCN and add predicted grades into df_grades(experiment_id.csv)
                rmse = gcn_run(experiment_id=experiment_id, params=param)

                gcn_rmses[str(control_friendship)].append(rmse)

        print(gcn_rmses)

        results_rows = []

        for model in MODELS:
            row = dict()
            row['model'] = model
            for control_friendship in control_friendships:

                if model == 'GCN':
                    row[control_friendship] = np.mean(gcn_rmses[str(control_friendship)])
                    print(gcn_rmses[str(control_friendship)])
                else:
                    experiments_info = load_experiment_infos()
                    row[control_friendship] = experiments_info[experiments_info['experiment_id'] == experiment_id][model+'_rmse'].iloc[0]

                print(row)

            results_rows.append(row)

        results = pd.DataFrame(results_rows).round(6)
        results.to_csv(STUDY_DIR + 'homophily_results.csv', index=False)
        print(results)
