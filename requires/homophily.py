import pandas as pd
from modelsControllers.simulated_data.gcn_pg_optuna_no_val import run as gcn_run
from requires.interpret_results import experiment_rmse
import sys
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from requires.config import ROOT_DIR, STUDY_DIR

experiment_ids = [1]

control_friendships = [0.0001, 0.001, 0.01, 0.1]


gcn_params = [
    {"epochs": 800, "dimension": 64, "learning_rate": 0.02, "number_of_layer": 3}
]


friends_count = []


def load_grades():

    df = pd.read_csv(STUDY_DIR + str(experiment_id)+'.csv')
    df['friends'] = ''
    return df


def update_friendships(_df_grades):
    return _df_grades.to_csv(STUDY_DIR + str(experiment_id)+'.csv', index=False)


def generate_friendships(_df_grades, control_friendship=None):

    global friends_count

    number_of_freinds = []
    for index, row in _df_grades.iterrows():
        grade = row['actual_grade']
        friends = _df_grades[(_df_grades['actual_grade'] <= grade + control_friendship) &
                          (_df_grades['actual_grade'] >= grade - control_friendship)]

        # print("control_friendship:", control_friendship, "friends.shape:", friends.shape)
        number_of_freinds.append(friends.shape[0])

        if not friends.empty:
            friends_list = list(friends.index)
            friends_list.remove(index)
            friends_str = ','.join([str(i) for i in friends_list])

        _df_grades.loc[index, 'friends'] = friends_str

    print("Avg number_of_freinds for :", control_friendship, np.mean(number_of_freinds))
    # sys.exit()

    friends_count_dict = dict()
    friends_count_dict[control_friendship] = np.mean(number_of_freinds)
    friends_count.append(friends_count_dict)

    return _df_grades


def run():
    run_model = True
    plot = True

    if run_model:

        for _experiment_id in experiment_ids:

            global experiment_id
            experiment_id = _experiment_id

            df_grades = load_grades()

            gcn_rmses = dict()

            results_rows = []
            for param in gcn_params:

                    for control_friendship in control_friendships:

                        gcn_rmses[str(control_friendship)] = []

                        for run in range(1, 3):

                            # generate friendships
                            df_grades = generate_friendships(df_grades, control_friendship=control_friendship)

                            # update experiment_id.csv file with new friendships
                            update_friendships(df_grades)

                            # run GCN and add predicted grades into df_grades(experiment_id.csv)
                            rmse = gcn_run(_experiment_id=experiment_id, _homophily_info={ "control_friendship": control_friendship, "run": run }, params=param)

                            # # load grades again, because it is updated by GCN
                            # df_grades = load_grades()
                            
                            # # interpret results
                            # gcn_rmses[str(control_friendship)] = experiment_rmse(_df_grades=df_grades, folds=[1, 2, 3, 4, 5])
                            
                            # # plot homophily study varying friendship t in |x1-x2| < t
                            gcn_rmses[str(control_friendship)].append(rmse)

                    # sys.exit()

                    print("gcn_rmses")
                    print(gcn_rmses)

                    results_rows = []
                    row = dict()
                    row['model'] = 'GCNSOAN'
                    for control_friendship in control_friendships:
                        row[control_friendship] = np.mean(gcn_rmses[str(control_friendship)])
                    
                    results_rows.append(row)

                    results = pd.DataFrame(results_rows).round(6)
                    results.to_csv(STUDY_DIR + 'homophily_results.csv', index=False)

                    print(results)



if __name__ == '__main__':
    run()
    

