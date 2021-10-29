import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

from requires.config import REAL_DATA_RAW_DIR


def load_data():
    _maxgrades = pd.read_csv(REAL_DATA_RAW_DIR + 'maxgrades.csv')
    _peer = pd.read_csv(REAL_DATA_RAW_DIR + 'peer.csv')
    _self = pd.read_csv(REAL_DATA_RAW_DIR + 'self.csv')
    _ta = pd.read_csv(REAL_DATA_RAW_DIR + 'ta.csv')

    return _maxgrades, _peer, _self, _ta


def normalise_grade(row, _maxgrades):
    return round(int(row['grade']) / int(_maxgrades[_maxgrades['taskid'] == row['taskid']]['maxgrade'].iloc[0]), 2)


def clean_data(_maxgrades, _peer, _self, _ta):
    _maxgrades = _maxgrades.astype(str)
    _peer = _peer.astype(str)
    _self = _self.astype(str)
    _ta = _ta.astype(str)

    # normalise grades
    _peer['grade'] = _peer.apply(lambda row: normalise_grade(row, _maxgrades), axis=1)
    _ta['grade'] = _ta.apply(lambda row: normalise_grade(row, _maxgrades), axis=1)
    _self['grade'] = _self.apply(lambda row: normalise_grade(row, _maxgrades), axis=1)

    return _maxgrades, _peer, _self, _ta


def get_group_members(taskid, groupid, _self):
    return _self[(_self['taskid'] == taskid) & (_self['groupid'] == groupid)]['reviewerid']


def add_assignment_id(_df):
    _df['assignmentid'] = _df['taskid'] + "-" + _df['groupid']

    return _df


def ignore_tasks(df, tasks=[]):

    df = df[~df['taskid'].isin(tasks)]

    return df


def run():
    maxgrades, peer, self, ta = load_data()
    maxgrades, peer, self, ta = clean_data(maxgrades, peer, self, ta)

    ta['assignment_id'] = ta['taskid'] + '-' + ta['groupid']

    print(ta.head(10))


def get_group_plot_id(ta, group_id):

    group_plot_id = ta[ta['groupid'] == group_id]['group_plot_id'].iloc[0]

    return group_plot_id


def plot_grades_by_student_id(task_ids=None):

    maxgrades, peer, self, ta = load_data()
    maxgrades, peer, self, ta = clean_data(maxgrades, peer, self, ta)

    ta = ta[ta['taskid'].isin(task_ids)]
    ta = ta.sort_values(by='grade')
    ta['groupid'] = ta['groupid'].astype(int)
    ta['group_plot_id'] = np.arange(0, ta.shape[0])

    peer = peer[peer['taskid'].isin(task_ids)]
    peer['groupid'] = peer['groupid'].astype(int)
    peer['group_plot_id'] = peer.apply(lambda row: get_group_plot_id(ta, row['groupid']), axis=1)
    peer = peer.sort_values(by='group_plot_id')

    plt.scatter(peer['group_plot_id'], peer['grade'].values, s=5, c='orange')
    plt.scatter(ta['group_plot_id'].values, ta['grade'].values, s=2, c='black')

    plt.title('Comparison of peer grades and ground truth grades', fontsize=10)

    plt.xlabel('Student ids sorted by ground truth grade')
    plt.ylabel('Grade')

    # plt.savefig("visualizes/simulated_data_analysis/peer-grades-skew+2-a0.25.png", dpi=300)

    plt.show()


def get_ta_grade(ta, group_id):

    ta_grade = ta[ta['groupid'] == group_id]['grade'].iloc[0]

    return ta_grade


def plot_grades_by_grade(task_ids=None):

    maxgrades, peer, self, ta = load_data()
    maxgrades, peer, self, ta = clean_data(maxgrades, peer, self, ta)

    ta = ta[ta['taskid'].isin(task_ids)]
    ta = ta.sort_values(by='grade')
    ta['groupid'] = ta['groupid'].astype(int)
    print(ta.head(20))

    peer = peer[peer['taskid'].isin(task_ids)]
    peer['groupid'] = peer['groupid'].astype(int)
    peer['ta_grade'] = peer.apply(lambda row: get_ta_grade(ta, row['groupid']), axis=1)
    peer = peer.sort_values(by='ta_grade')

    print(peer.head(20))
    # sys.exit()

    plt.scatter(peer['ta_grade'].values, peer['grade'].values, s=5, c='orange')
    plt.scatter(ta['grade'].unique(), ta['grade'].unique(), s=1, c='black')

    plt.title('Comparison of peer grades and ground truth grades', fontsize=10)

    plt.xlabel('Student ids sorted by ground truth grade')
    plt.ylabel('Grade')

    # plt.savefig("visualizes/simulated_data_analysis/peer-grades-skew+2-a0.25.png", dpi=300)

    plt.show()


if __name__ == '__main__':
    plot_grades_by_student_id(task_ids=['43'])
    plot_grades_by_grade(task_ids=['43'])
