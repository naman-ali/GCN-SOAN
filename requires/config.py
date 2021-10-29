import numpy as np
import os
import sys
from set_study import study, custom_study
import shutil


# Simulated data global varibales
studies = {
    1: 'number_of_graders',
    2: 'graders_bias',
    3: 'ground_truth_distribution',
    4: 'working_impact_grading',
    5: 'erdos',
    6: 'homophily',
    7: 'real_data_peer_evaluation',
    8: 'real_data_peer_and_self_evaluation',
    9: 'custom'
}


studies_names = list(studies.values())
studies_numbers = list(studies.keys())

STUDY_NUMBER = studies_numbers[studies_names.index(study)]

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/"
ROOT_DIR = ROOT_DIR.replace('requires/', '')

STUDY_DIR = ROOT_DIR + 'data/simulated_studies/study' + str(STUDY_NUMBER) + '/'


if os.path.isdir(STUDY_DIR) and STUDY_NUMBER != 6:
    shutil.rmtree(STUDY_DIR)
if not os.path.isdir(STUDY_DIR):
    os.makedirs(STUDY_DIR)

EXPERIMENT_INFO_FILE = ROOT_DIR + 'data/simulated_studies/study' + str(STUDY_NUMBER) + '/experiments_info.csv'

# 1 - Number of graders
if study == 'number_of_graders':

    class_sizes = [500]
    number_of_graders = range(2, 11)

    locs1 = [0.3]
    scales1 = [0.1]
    left_pick_portion = 1  # means 0, 1, 2, 3, 4

    locs2 = [0.7]
    scales2 = [0.1]

    grader_error_type = 'constant_std'  # constant_std, working_impact_grading
    control_grader_errors = [0.25]  # std: constant_std | beta: working_impact_grading

    friendship_type = 'none'  # erdos, homophily, none
    control_friendships = [0.0]  # erdos probability | t in |gi-gj| <= t : [0.01, 0.05, 0.1, 0.2, 0.5, 1]

    graders_behaviours = [{
        'alpha': [round(i, 1)],
        'probabilties': [1]
    } for i in np.arange(0, 0.1, 0.1)]

    skewness_method = 'biased'  # biased, azzalini

    study_title = 'Study 1: number_of_graders'
    x_axis = 'number_of_graders'


# 2 - Graders bias
if study == 'graders_bias':

    class_sizes = [500]  # 3 items
    number_of_graders = range(3, 4)  # 3 items

    locs1 = [0.3]
    scales1 = [0.1]
    left_pick_portion = 1

    locs2 = [0.7]
    scales2 = [0.1]

    grader_error_type = 'constant_std'  # constant_std, working_impact_grading
    control_grader_errors = [0.25]  # std: constant_std | beta: working_impact_grading

    friendship_type = 'none'  # erdos, homophily, none
    control_friendships = [0.0]  # erdos probability | t in |gi-gj| <= t : [0.01, 0.05, 0.1, 0.2, 0.5, 1]

    graders_behaviours = [{
        'alpha': [round(i, 1)],
        'probabilties': [1]
    } for i in [-0.3, -0.15, 0.0, 0.15, 0.3]]

    skewness_method = 'biased'  # biased, azzalini

    study_title = 'Study 2: graders_bias'
    x_axis = 'alpha'

# 3 - Ground truth distribution
if study == 'ground_truth_distribution':

    class_sizes = [500]  # 3 items
    number_of_graders = range(3, 4)  # 3 items

    locs1 = [0.3]
    scales1 = [0.1]
    left_pick_portion = 0

    locs2 = np.arange(0.1, 1.0, 0.1)
    scales2 = [0.15]

    grader_error_type = 'constant_std'  # constant_std, working_impact_grading
    control_grader_errors = [0.25]  # std: constant_std | beta: working_impact_grading

    friendship_type = 'none'  # erdos, homophily, none
    control_friendships = [0.0]  # erdos probability | t in |gi-gj| <= t : [0.01, 0.05, 0.1, 0.2, 0.5, 1]


    graders_behaviours = []
    for i in np.arange(0, 0.1, 0.1):
        i = round(i, 1)
        graders_behaviours.append({
            'alpha': [i],
            'probabilties': [1]
        })
        print(i)
    skewness_method = 'biased'  # biased, azzalini

    study_title = 'Study 3: ground_truth_distribution'
    x_axis = 'mean2'

# 4 - working_impact_grading
if study == 'working_impact_grading':

    class_sizes = [500]  # 3 items
    number_of_graders = range(3, 4)  # 3 items

    locs1 = [0.3]
    scales1 = [0.1]
    left_pick_portion = 1

    locs2 = [0.7]
    scales2 = [0.1]

    grader_error_type = 'working_impact_grading_bt'  # constant_std, working_impact_grading, working_impact_grading_bt
    control_grader_errors = np.arange(0.2, 1.2, 0.2)  # std: constant_std | beta: working_impact_grading

    friendship_type = 'none'  # erdos, homophily, none
    control_friendships = [0.0]  # erdos probability | t in |gi-gj| <= t : [0.01, 0.05, 0.1, 0.2, 0.5, 1]


    graders_behaviours = []
    for i in np.arange(0, 0.1, 0.1):
        i = round(i, 1)
        graders_behaviours.append({
            'alpha': [i],
            'probabilties': [1]
        })
    skewness_method = 'biased'  # biased

    study_title = 'Study 4: working_impact_grading'
    x_axis = 'beta'

# 5 - Erdos study
if study == 'erdos':

    class_sizes = [500]
    number_of_graders = [3]
    locs1 = [0.3]
    scales1 = [0.1]
    left_pick_portion = 1  # 100, 400
    locs2 = [0.7]
    scales2 = [0.1]
    grader_error_type = 'constant_std'  # constant_std, working_impact_grading
    control_grader_errors = [0.25]  # std: constant_std | beta: working_impact_grading
    friendship_type = 'erdos'  # erdos, homophily, none
    control_friendships = [0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 1]  # erdos probability | t in |gi-gj| <= t
    graders_behaviours = [{
            'alpha': [round(i, 1)],
            'probabilties': [1]
        } for i in np.arange(0, 0.1, 0.1)]
    skewness_method = 'biased'
    study_title = 'Study 5: erdos'
    x_axis = 'control_friendship'

# 6 - Homophily study
if study == 'homophily':

    class_sizes = [500]  # 3 items
    number_of_graders = [3]  # 3 items

    locs1 = [0.3]
    scales1 = [0.1]
    left_pick_portion = 1

    locs2 = [0.7]
    scales2 = [0.1]

    grader_error_type = 'constant_std'  # constant_std, working_impact_grading
    control_grader_errors = [0.25]  # std: constant_std | beta: working_impact_grading

    friendship_type = 'homophily'  # erdos, homophily, none
    control_friendships = [0.0001, 0.001, 0.01, 0.1]

    graders_behaviours = []
    for i in np.arange(0, 0.1, 0.1):
        i = round(i, 1)
        graders_behaviours.append({
            'alpha': [i],
            'probabilties': [1]
        })
    skewness_method = 'biased'  # biased, azzalini

    study_title = 'Study 6: homophily'
    x_axis = 'alpha'


# 9 - Custom study
if study == 'custom':

    class_sizes = [custom_study['class_size']]
    number_of_graders = [custom_study['number_of_graders']]

    locs1 = [custom_study['left_peak_mean']]
    scales1 = [custom_study['left_peak_std']]
    if custom_study['distribution'] == 'bimodal':
        left_pick_portion = 1
    else:
        left_pick_portion = 0

    locs2 = [custom_study['right_peak_mean']]
    scales2 = [custom_study['right_peak_std']]

    grader_error_type = 'constant_std'  # constant_std, working_impact_grading
    control_grader_errors = [custom_study['graders_std']]  # std: constant_std | beta: working_impact_grading

    friendship_type = 'none'  # erdos, homophily, none
    control_friendships = [0.0]  # erdos probability | t in |gi-gj| <= t : [0.01, 0.05, 0.1, 0.2, 0.5, 1]

    graders_behaviours = [{
        'alpha': [custom_study['graders_bias']],
        'probabilties': [1]
        }
    ]

    skewness_method = 'biased'

    study_title = 'Study 9: custom_study'
    x_axis = 'number_of_graders'




# Real data global variables
REAL_DATA_RAW_DIR = ROOT_DIR + 'data/real_data_raw/'
REAL_DATA_PROCESSED_DIR = ROOT_DIR + 'data/real_data_processed/'


runs = 3
MODELS = [
    'Average',
    'Median',
    'GCN',
    'PeerRank',
    'RankwithTA',
    'Vancouver',
    'TunedModel'
]
max_grade = 1

fast_gcn = False