from modelsControllers.real_data.TunedModel import TunedModel
from modelsControllers.real_data.PeerRank import PeerRank
from modelsControllers.real_data.RankwithTA import RankwithTA
from modelsControllers.real_data.vancouver import run as vancouver_run
from requires.interpret_results_real_data import run as interpret_results
from modelsControllers.real_data.gcn_soan_real_data import run as gcn_run
from requires.helper import *
from requires import gg_creator


REAL_DATA_PROCESSED_DIR = ROOT_DIR + 'data/real_data_processed/'


def load_gg():
    _gg_df = pd.read_csv('data/AD/processed/group-group.csv')
    _gg_df = _gg_df.drop(columns=['Unnamed: 0'])
    _gg_df = _gg_df.round(1)


    return _gg_df


def load_predicted_grades():
    return pd.read_csv(REAL_DATA_PROCESSED_DIR + 'gg-models-results.csv')


def create_predicted_grade_df(gg_df):
    groups = gg_df.groupby(['group_id', 'task_id'])

    df_array = []
    for name, group in groups:
        peer_group = group[group['grader_type'] == 'peer']

        self_group = group[group['grader_type'] == 'self']
        self_grade = np.nan
        if not self_group.empty:
            self_grade = self_group['grade'].iloc[0]

        ta_group = group[group['grader_type'] == 'TA']
        ta_grade = np.nan
        if len(ta_group) != 0:
            ta_grade = ta_group['grade'].iloc[0]

        row = {
            'group_id': name[0],
            'task_id': name[1],
            'peer_graders': implode(peer_group['grader_group_id'].tolist()),
            'peer_grades': implode(peer_group['grade'].tolist()),
            'TA_grade': ta_grade,
            'self_grade': self_grade
        }

        df_array.append(row)

    df_models = pd.DataFrame(df_array)

    df_models.to_csv(REAL_DATA_PROCESSED_DIR + 'gg-models-results.csv', index=False)

    return df_models


def run_models(gg_df, models_df, self_evaluation=False, models=None):

    TA_df = gg_df[gg_df['grader_type'] == 'TA']

    if self_evaluation:
        gg_df = gg_df[gg_df['grader_type'] != 'TA']
    else:
        gg_df = gg_df[gg_df['grader_type'] == 'peer']

    gg_df = gg_df[~gg_df['task_id'].isin(['51', '52', '61', '62'])]

    for name, group in gg_df.groupby('task_id'):

        print("--- Run baselines on task id:", name)

        # run TunedModel model
        if 'TunedModel' in models:
            print("Running tunedModel...")
            TunedModel(group)

        # run PeerRank model
        if 'PeerRank' in models:
            print("Running PeerRank...")
            PeerRank(group)

        # run RankwithTA()
        if 'RankwithTA' in models:
            print("Running RankwithTA...")
            for fold in [1, 2, 3, 4]:
                RankwithTA(group, fold=fold)

        # run PeerRank model
        if 'Vancouver' in models:
            print("Running Vancouver...")
            vancouver_run(df=group)

    models_df = pd.read_csv(REAL_DATA_PROCESSED_DIR + 'gg-models-results.csv')

    # avg grade
    if 'Average' in models:
        for index, row in models_df.iterrows():
            grades = [float(i) for i in row['peer_grades'].split(',')]
            models_df.at[index, 'Average_grade'] = np.mean(grades)

    # median grade
    if 'Median' in models:
        for index, row in models_df.iterrows():
            grades = [float(i) for i in row['peer_grades'].split(',')]
            models_df.at[index, 'Median_grade'] = np.median(grades)

    models_df.to_csv(REAL_DATA_PROCESSED_DIR + 'gg-models-results.csv', index=False)

    return models_df


def append_self_grades_to_peer_grades(model_results):

    model_results['peer_graders'] = model_results['peer_graders'] + "," + model_results['group_id'].astype(str)
    model_results['peer_grades'] = model_results['peer_grades'] + "," + model_results['self_grade'].astype(str)

    model_results.to_csv(REAL_DATA_PROCESSED_DIR + 'gg-models-results.csv', index=False)

    return model_results


def run_baselines(gg_df=None, predicted_grades=None, self_evaluation=False):

    # for self evaluation study
    if self_evaluation:
        predicted_grades = append_self_grades_to_peer_grades(predicted_grades)
        predicted_grades.to_csv(REAL_DATA_PROCESSED_DIR + 'gg-models-results.csv', index=False)

    models_df = run_models(gg_df, predicted_grades, self_evaluation=self_evaluation, models=MODELS)

    return models_df


def load_rmse_models():

    df = pd.read_csv(REAL_DATA_PROCESSED_DIR + 'rmse_models.csv')

    return df


def reset_files():
    from pathlib import Path

    files = ['group-group.csv', 'gg-models-results.csv', 'RankwithTA.csv']

    for file in files:
        my_file = Path(REAL_DATA_PROCESSED_DIR + file)
        if my_file.is_file():
            os.remove(REAL_DATA_PROCESSED_DIR + file)

    return True


def run():

    reset_files()

    # gcn-soan
    gcn_rmses = gcn_run()

    print("------------ gcn_rmses ------------")
    print(gcn_rmses)

    # baselines
    self_evaluation = False
    if study == 'real_data_peer_and_self_evaluation':
        self_evaluation = True

    gg_creator.run()
    gg_df = pd.read_csv(REAL_DATA_PROCESSED_DIR + 'group-group.csv')

    predicted_grades = create_predicted_grade_df(gg_df)

    # for self evaluation study
    if self_evaluation:
        predicted_grades = append_self_grades_to_peer_grades(predicted_grades)

    predicted_grades = load_predicted_grades()

    models_df = run_baselines(self_evaluation=self_evaluation, gg_df=gg_df, predicted_grades=predicted_grades)

    # interpret results
    rmse_models = interpret_results(gcn_rmses=gcn_rmses)

    rmse_models['best'] = rmse_models.idxmin(axis=1)
    print("GCN rmses:", gcn_rmses)
    print(rmse_models)


