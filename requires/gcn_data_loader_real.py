from requires import real_data_helper
import pandas as pd
import numpy as np
import os
from sklearn import model_selection
from requires.config import REAL_DATA_PROCESSED_DIR



def load_simulated_G():
    _G = pd.read_csv('../data/simulated/G-1-20.csv', index_col=False)

    # _G = _G.drop(columns=['actual_grade'])

    return _G


def create_G(taskid=None, ownership=False, self_evaluation=False, peer_evaluation=False):
    _maxgrades, _peer, _self, _ta = real_data_helper.load_data()
    _maxgrades, _peer, _self, _ta = real_data_helper.clean_data(_maxgrades, _peer, _self, _ta)

    if peer_evaluation:
        _peer['grade'] = _peer['grade'].astype(float)
        _peer['grade'] = _peer['grade']
        _peer = real_data_helper.add_assignment_id(_peer)
        _peer = real_data_helper.ignore_tasks(_peer, ['51', '52', '61', '62'])
        if taskid != "All":
            _peer = _peer[
                (_peer['taskid'].isin(taskid))
            ]

    if self_evaluation:
        # self evaluation grades dataframe
        _self['grade'] = _self['grade'].astype(float)
        _self['grade'] = _self['grade']
        _self = real_data_helper.add_assignment_id(_self)
        _self = real_data_helper.ignore_tasks(_self, ['51', '52', '61', '62'])
        if taskid != "All":
            _self = _self[
                (_self['taskid'].isin(taskid))
            ]

    _G = pd.DataFrame(columns=['source', 'target', 'edge_type', 'weight'])

    if peer_evaluation:
        # adding peer grades edges
        peer_grades_edges = pd.DataFrame()
        peer_grades_edges['source'] = _peer['reviewerid']
        peer_grades_edges['target'] = _peer['assignmentid']

        peer_grades_edges['edge_type'] = 'grade'
        peer_grades_edges['weight'] = _peer['grade']

        _G = pd.concat([_G, peer_grades_edges], ignore_index=True)

    if self_evaluation:
        # adding self evaluation grades edges
        self_grades_edges = pd.DataFrame()
        self_grades_edges['source'] = _self['reviewerid']
        self_grades_edges['target'] = _self['assignmentid']

        self_grades_edges['edge_type'] = 'grade'

        weight = _self['grade']
        if self_evaluation & ownership:
            weight = _self['grade'] + 1
        self_grades_edges['weight'] = weight

        _G = pd.concat([_G, self_grades_edges], ignore_index=True)

    if ownership & (not self_evaluation):
        # create ownership edges: student(A) <---> assignment(B) means assignment(B) is done by student(A)
        df_array = []

        if peer_evaluation:
            _groups_checker = _peer

        if self_evaluation:
            _groups_checker = _self

        for label, group in _groups_checker.groupby(['taskid', 'groupid']):
            group_members = real_data_helper.get_group_members(label[0], label[1], _self)
            for member in group_members:
                row = dict()
                row['source'] = member
                row['target'] = label[0] + '-' + label[1]
                row['weight'] = 1
                row['edge_type'] = 'ownership'
                df_array.append(row)

        ownership = pd.DataFrame(df_array)

        _G = pd.concat([_G, ownership], ignore_index=True)

    return _G


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


def get_labels(_G, dummies=True, student_labels=False):
    _maxgrades, _peer, _self, _ta = real_data_helper.load_data()
    _maxgrades, _peer, _self, _ta = real_data_helper.clean_data(_maxgrades, _peer, _self, _ta)

    _ta = real_data_helper.add_assignment_id(_ta)

    _labels = pd.DataFrame()

    if student_labels:
        all_nodes = np.concatenate((_G['source'].unique(), _G['target'].unique()), axis=None)
        _labels['node'] = all_nodes
    else:
        _labels['node'] = _G['target'].unique()

    _labels['label'] = _labels.apply(lambda row: get_node_label(row['node'], _ta), axis=1)
    # _labels['label'] = _labels.apply(lambda row: get_average_grade(row['node'], _G), axis=1)

    # _labels = _labels.set_index('node')

    _labels['label'] = _labels['label'].astype(str)

    # _labels.index.name = None

    # _labels.loc[_labels['label'] == '0.11', 'label'] = '0.1'
    # _labels.loc[_labels['label'] == '0.09', 'label'] = '0.1'

    _labels['label'] = _labels['label'].astype(float)
    _labels['label'] = _labels['label']
    # _labels['label'] = _labels['label'].astype(str)

    _labels.loc[_labels['label'] != -1, 'label'] = _labels['label'] + 0

    # _labels = _labels[_labels['label'] != 0]

    # print(_labels.value_counts())

    if dummies:
        _labels = pd.get_dummies(_labels, prefix='')

    return _labels


def get_feature(node, _peer):
    if '-' in node:
        query = _peer[_peer['assignmentid'] == node]
        return ','.join(query['grade'].tolist())
    else:
        query = _peer[_peer['reviewerid'] == node]
        return ','.join(query['grade'].tolist())


def get_nodes_features(_G):
    _maxgrades, _peer, _self, _ta = real_data_helper.load_data()
    _maxgrades, _peer, _self, _ta = real_data_helper.clean_data(_maxgrades, _peer, _self, _ta)

    _peer = real_data_helper.add_assignment_id(_peer)
    _peer['grade'] = _peer['grade'].astype(str)

    df_array = []
    for node in _G['source'].unique():
        grades = _peer[_peer['reviewerid'] == node]['grade']

        row = dict()
        row['node_type'] = 0


    _features = pd.DataFrame()
    _features['string_target'] = _G['target'].apply(lambda node: get_feature(node, _peer))
    _features['string_source'] = _G['source'].apply(lambda node: get_feature(node, _peer))


def get_grades(node, _G):
    return _G[_G['target'] == node]['weight'].tolist()


def get_graders(node, _G):
    return _G[_G['target'] == node]['source'].tolist()


def get_node_id(nodes_df, node_name):
    node_df = nodes_df[nodes_df['name'] == node_name]

    if node_df.empty:
        node_id = np.nan
    else:
        node_id = node_df['id'].iloc[0]

    return node_id


def assignid_to_assignnumber(assignid):
    taskid = assignid.split('-')[0]
    assignnumber = taskid[0]

    return assignnumber


def save_fold(foldid=None, train=None, test=None):

    assignment = assignid_to_assignnumber(train.node.values[0])

    train = train.drop(columns=['id'])
    test = test.drop(columns=['id'])

    train.columns = ['assignid', 'label']
    test.columns = ['assignid', 'label']

    train = train.set_index('assignid')
    test = test.set_index('assignid')


    if not os.path.isdir(REAL_DATA_PROCESSED_DIR + 'folds'):
        os.makedirs(REAL_DATA_PROCESSED_DIR + 'folds')

    train.to_csv(REAL_DATA_PROCESSED_DIR + 'folds/assign{}-fold{}-train.csv'.format(assignment, foldid), index_label='assignid')
    test.to_csv(REAL_DATA_PROCESSED_DIR + 'folds/assign{}-fold{}-test.csv'.format(assignment, foldid), index_label='assignid')


def split_train_test(assignment_labels=None, test_size=None):
    boot = model_selection.ShuffleSplit(n_splits=4, test_size=test_size, random_state=42)

    assignment_nodes_ids = assignment_labels['id'].values

    folds = dict()
    fold_id = 1
    for train, test in boot.split(assignment_nodes_ids):
        labels_df_test = assignment_labels[assignment_labels['id'].isin(assignment_nodes_ids[test])]
        labels_df_train = assignment_labels[assignment_labels['id'].isin(assignment_nodes_ids[train])]

        folds[str(fold_id)] = {
            'train': labels_df_train['id'].values,
            'test': labels_df_test['id'].values
        }

        save_fold(foldid=fold_id, train=labels_df_train, test=labels_df_test)

        fold_id = fold_id + 1

    return folds


def get_real_data(taskid='all',
                  assignment='all',
                  features='one',
                  ownership=False,
                  self_evaluation=False,
                  peer_evaluation=False,
                  test_size=0.8):

    G_df = create_G(taskid=taskid, ownership=ownership, self_evaluation=self_evaluation,
                    peer_evaluation=peer_evaluation)

    # get node names (student nodes, and assignment nodes)
    node_names = pd.concat([G_df['source'], G_df['target']]).unique()

    nodes_df = pd.DataFrame()
    nodes_df['name'] = node_names
    nodes_df['id'] = nodes_df.index

    G_df['source_id'] = G_df.apply(lambda row: get_node_id(nodes_df, row['source']), axis=1)
    G_df['target_id'] = G_df.apply(lambda row: get_node_id(nodes_df, row['target']), axis=1)

    labels = get_labels(G_df, dummies=False, student_labels=True)

    labels['id'] = labels.apply(lambda row: get_node_id(nodes_df, row['node']), axis=1)

    features = np.ones((G_df['source'].unique().shape[0] + G_df['target'].unique().shape[0], 10))

    assignment_labels = labels[labels['label'] != -1]
    folds = split_train_test(assignment_labels=assignment_labels, test_size=test_size)

    return features, labels['label'].values, G_df, folds


def split_chr(word):
    return [char for char in word]


if __name__ == '__main__':
    print("main")
