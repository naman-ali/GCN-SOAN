import pandas as pd
import sys
import numpy as np
import statistics
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import preprocessing, model_selection
from sklearn.model_selection import train_test_split
pd.options.mode.chained_assignment = None  # default='warn'
mpl.rcParams['figure.dpi'] = 300
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
import os

ROOT_DIR = os.getcwd() + '/'

# ROOT_DIR = '/Applications/Coding/Data_Science/UOIT/skeleton/'


grade_round = 2
# weighted = True
# grades_as = 'edge_type'
grades_as = 'edge_weight'
membership_type = False

# grades_as = 'no_type_and_weight'



def load_simulated_G():
    _G = pd.read_csv(ROOT_DIR+'data/simulated/G-1-20.csv', index_col=False)

    # _G = _G.drop(columns=['actual_grade'])

    return _G


def load_simulated_L():
    _L = pd.read_csv(ROOT_DIR+'data/simulated/L-1-20.csv', index_col='id')

    _L.columns = ['label']

    _L['label'] = round(_L['label']/100, grade_round)

    return _L


def create_G(simulated_df=None, experiment_id=None, ownership=False, friendship=False):

    simulated_df['friends'] = simulated_df['friends'].astype(str)

    max_index = simulated_df.index.max()

    df_array = []

    # for each student generate peer grades and friendship edges
    for student_id, row in simulated_df.iterrows():

        graders = row['graders'].split(',')
        grades = row['grades'].split(',')

        friends = row['friends'].split(',')

        # peer grade edges
        for index, grader in enumerate(graders):
            new_row = dict()
            new_row['source'] = int(grader)
            new_row['target'] = int(student_id)
            new_row['edge_type'] = 'grade'
            new_row['weight'] = float(grades[index])
            new_row['source_id'] = int(grader)
            new_row['target_id'] = int(student_id) + int(max_index) + 1

            df_array.append(new_row)

        # ownership edges
        if ownership:
            new_row = dict()
            new_row['source'] = int(student_id)
            new_row['target'] = int(student_id)
            new_row['edge_type'] = 'ownership'
            new_row['weight'] = 1
            new_row['source_id'] = int(student_id)
            new_row['target_id'] = int(student_id) + int(max_index) + 1

            df_array.append(new_row)

        # friendship edges
        if friendship and friends and not 'nan' in friends:
            for index, friend in enumerate(friends):
                new_row = dict()
                new_row['source'] = int(friend)
                new_row['target'] = int(student_id)
                new_row['edge_type'] = 'friendship'
                new_row['weight'] = 1
                new_row['source_id'] = int(friend)
                new_row['target_id'] = int(student_id)

                df_array.append(new_row)


    _G = pd.DataFrame(df_array)

    # drop duplicate friendship edges
    if friendship:
        _G_f = _G[_G['edge_type'] == 'friendship']
        _G_f['ordered-name'] = _G_f.apply(lambda x: '-'.join(sorted([str(x['source_id']), str(x['target_id'])])), axis=1)
        duplicates = _G_f[_G_f.duplicated(subset=['ordered-name'])]
        _G = _G.drop(duplicates.index)

    _G = _G.sort_values(by='source')

    # scale grades between x and y
    _G['weight'] = _G['weight'] + 0

    _G['weight'] = _G['weight'].round(grade_round)


    return _G


def get_ta_grade(assignment_id, _ta):
    query = _ta[_ta['assignmentid'] == assignment_id]['grade']
    if len(query) == 0:
        print("TA grade is missing for assignment id: ", assignment_id)
        return 0.6

    return query.iloc[0]


def get_node_label(node, _ta):

    if '-' not in node:
        label = -1
    else:
        query = _ta[_ta['assignmentid'] == node]['grade']
        label = query.iloc[0]
        if len(query) == 0:
            print("TA grade is missing for assignment id: ", node)
            label = 0.6

    return label


def get_actual_grade(simulated_df, student_id):

    actual_grade = simulated_df.loc[[student_id]]['actual_grade'].iloc[0]

    return actual_grade


def get_labels(G_df=None, simulated_df=None):

    df_array = []

    graders_nodes = pd.DataFrame()
    graders_nodes['node_id'] = G_df['source_id'].unique()
    graders_nodes['node_name'] = G_df['source'].unique()
    graders_nodes['label'] = -1

    G_df = G_df.sort_values(by='target_id')
    assignment_nodes = pd.DataFrame()
    assignment_nodes['node_id'] = G_df[G_df['edge_type'] == 'grade']['target_id'].unique()
    assignment_nodes['node_name'] = G_df[G_df['edge_type'] == 'grade']['target'].unique()
    assignment_nodes['label'] = assignment_nodes.apply(lambda row: get_actual_grade(simulated_df, row['node_name']), axis=1)

    _labels = pd.concat([graders_nodes, assignment_nodes])

    _labels['label'] = _labels['label'].round(grade_round)

    _labels['label'] = _labels['label'] + 0

    return _labels


def get_features(G_df=None, features=None):

    if features == 'one-hot':
        features = np.identity(G_df['source'].unique().shape[0] + G_df['target'].unique().shape[0])
    else:
        features = np.ones((G_df['source'].unique().shape[0] + G_df['target'].unique().shape[0], 10))

    return features


def get_feature(node, _peer):
    if '-' in node:
        query = _peer[_peer['assignmentid'] == node]
        return ','.join(query['grade'].tolist())
    else:
        query = _peer[_peer['reviewerid'] == node]
        return ','.join(query['grade'].tolist())


def get_nodes_features(_G):
    _maxgrades, _peer, _self, _ta = ad_helper.load_data()
    _maxgrades, _peer, _self, _ta = ad_helper.clean_data(_maxgrades, _peer, _self, _ta)

    _peer = ad_helper.add_assignment_id(_peer)
    _peer['grade'] = _peer['grade'].astype(str)

    df_array = []
    for node in _G['source'].unique():
        grades = _peer[_peer['reviewerid'] == node]['grade']

        row = dict()
        row['node_type'] = 0
        # for grade in grades:
        #     row[]

    _features = pd.DataFrame()
    _features['string_target'] = _G['target'].apply(lambda node: get_feature(node, _peer))
    _features['string_source'] = _G['source'].apply(lambda node: get_feature(node, _peer))


def get_grades(node, _G):
    return _G[_G['target'] == node]['weight'].tolist()


def get_graders(node, _G):
    return _G[_G['target'] == node]['source'].tolist()


def get_grade_and_grader(node, _G):
    graders = _G[(_G['target'] == node) & (_G['edge_type'] != 'membership')]['source'].tolist()

    if grades_as == 'edge_weight':
        grades = _G[(_G['target'] == node) & (_G['edge_type'] != 'membership')]['weight'].tolist()

    if grades_as == 'edge_type':
        grades = _G[(_G['target'] == node) & (_G['edge_type'] != 'membership')]['edge_type'].tolist()
        grades = [float(e) for e in grades if 'membership' not in e]

    owners = _G[(_G['target'] == node) & (_G['edge_type'] == 'membership')]['source'].tolist()

    # print(grades)
    # sys.exit()


    return (graders, grades, owners)


def get_average_grade(node, _G):
    graders = _G[(_G['target'] == node) & (_G['edge_type'] != 'membership')]['weight'].tolist()

    if grades_as == 'edge_weight':
        grades = _G[(_G['target'] == node) & (_G['edge_type'] != 'membership')]['weight'].tolist()

    if grades_as == 'edge_type':
        grades = _G[(_G['target'] == node) & (_G['edge_type'] != 'membership')]['edge_type'].tolist()
        grades = [float(e) for e in grades if 'membership' not in e]

    owners = _G[(_G['target'] == node) & (_G['edge_type'] == 'membership')]['source'].tolist()

    # print(node)
    # print(grades)
    # sys.exit()

    return np.mean(grades)


def get_median_grade(node, _G):
    graders = _G[(_G['target'] == node) & (_G['edge_type'] != 'membership')]['source'].tolist()

    if grades_as == 'edge_weight':
        grades = _G[(_G['target'] == node) & (_G['edge_type'] != 'membership')]['weight'].tolist()

    if grades_as == 'edge_type':
        grades = _G[(_G['target'] == node) & (_G['edge_type'] != 'membership')]['edge_type'].tolist()
        grades = [float(e) for e in grades if 'membership' not in e]

    owners = _G[(_G['target'] == node) & (_G['edge_type'] == 'membership')]['source'].tolist()

    return np.median(grades)


def get_grades(node, _G):

    if grades_as == 'edge_weight':
        grades = _G[(_G['target'] == node) & (_G['edge_type'] != 'membership')]['weight'].tolist()

    if grades_as == 'edge_type':
        grades = _G[(_G['target'] == node) & (_G['edge_type'] != 'membership')]['edge_type'].tolist()
        grades = [float(e) for e in grades if 'membership' not in e]

    grades = [str(round(e, grade_round)) for e in grades]

    return ','.join(grades)


def get_assign_id(taskid):

    assignid_splited = [int(i) for i in taskid[0]]
    assignid = assignid_splited[0]

    return assignid



def split_train_test(labels_df=None, test_size=None):


    assignment_node = labels_df[labels_df['node_id'] >= labels_df.shape[0]/2]

    boot = model_selection.ShuffleSplit(n_splits=4, test_size=test_size, random_state=42)

    assignment_nodes = assignment_node['node_id'].values

    folds = dict()
    fold_id = 1
    for train, test in boot.split(assignment_nodes):

        labels_df_test = labels_df[labels_df['node_id'].isin(assignment_nodes[test])]

        labels_df_train = labels_df[labels_df['node_id'].isin(assignment_nodes[train])]

        folds[fold_id] = {
            'train': labels_df_train['node_id'].values,
            # 'validation': labels_df_validation['node_id'].values,
            'test': labels_df_test['node_id'].values
        }

        fold_id = fold_id + 1

    return folds



def get_simulated_data(experiment_id=None, features='one', ownership=False, test_size=None, friendship=False):

    simulated_df = get_simulated_df(experiment_id=experiment_id)

    G_df = create_G(simulated_df=simulated_df, experiment_id=experiment_id, ownership=ownership, friendship=friendship)

    labels_df = get_labels(G_df=G_df, simulated_df=simulated_df)

    features = get_features(G_df=G_df, features=features)

    folds = split_train_test(labels_df=labels_df, test_size=test_size)

    return features, labels_df['label'].values, G_df, folds



def split_chr(word):
    return [char for char in word]


def get_simulated_df(experiment_id=None):

    df = pd.read_csv(ROOT_DIR+'data/simulated_studies/study6/{}.csv'.format(experiment_id))

    return df


if __name__ == '__main__':

    get_simulated_data(experiment_id=1, features='identity', ownership=False)

