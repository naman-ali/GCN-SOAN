import pandas as pd
import numpy as np
from requires import helper
from requires.config import REAL_DATA_RAW_DIR, ROOT_DIR

REAL_DATA_PROCESSED_DIR = ROOT_DIR + 'data/real_data_processed/'


def create_temp():
    # load 4 main files of AD dataset
    df_ta = pd.read_csv(REAL_DATA_RAW_DIR + 'ta.csv')
    df_peer = pd.read_csv(REAL_DATA_RAW_DIR + 'peer.csv')
    df_self = pd.read_csv(REAL_DATA_RAW_DIR + 'self.csv')
    df_max = pd.read_csv(REAL_DATA_RAW_DIR + 'maxgrades.csv')

    student_ids = np.sort(df_peer['reviewerid'].unique()).tolist()

    df_array = []
    for student_id in student_ids:

        group_id = df_self[df_self['reviewerid'] == student_id]['groupid'].unique()
        if group_id.size == 0:
            continue

        group_id = group_id[0]

        # add TA grades
        ta_grades = df_ta[df_ta['groupid'] == group_id]
        df_row = {}
        for index, row in ta_grades.iterrows():
            df_row = {
                'student_id': student_id,
                'group_id': group_id,
                'grader_id': row['reviewerid'],
                'grader_type': 'TA',
                'sheet': helper.split_int(row['taskid'])[0],
                'task': helper.split_int(row['taskid'])[1],
                'grade': round(row['grade'] / helper.get_max_grade(df_max, row['taskid']), 3)
            }
            df_array.append(df_row)


        # add peer grades
        ta_peer = df_peer[df_peer['groupid'] == group_id]
        df_row = {}
        for index, row in ta_peer.iterrows():
            df_row = {
                'student_id': student_id,
                'group_id': group_id,
                'grader_id': row['reviewerid'],
                'grader_type': 'peer',
                'sheet': helper.split_int(row['taskid'])[0],
                'task': helper.split_int(row['taskid'])[1],
                'grade': round(row['grade'] / helper.get_max_grade(df_max, row['taskid']), 3)
            }
            df_array.append(df_row)



        # add self grades
        ta_self = df_self[df_self['groupid'] == group_id]
        df_row = {}
        for index, row in ta_self.iterrows():
            df_row = {
                'student_id': student_id,
                'group_id': group_id,
                'grader_id': row['reviewerid'],
                'grader_type': 'self',
                'sheet': helper.split_int(row['taskid'])[0],
                'task': helper.split_int(row['taskid'])[1],
                'grade': round(row['grade'] / helper.get_max_grade(df_max, row['taskid']), 3)
            }
            df_array.append(df_row)


    df = pd.DataFrame(df_array)

    return df


def add_ta_grade(df):

    for index, row in df.iterrows():
        ta_grade = df[(df['student_id'] == row['student_id']) & (df['sheet'] == row['sheet']) & (df['task'] == row['task']) & (df['grader_type'] == 'TA')]['grade'].iloc[0]
        df.at[index, 'ta-grade'] = round(ta_grade, 3)

    return df


def make_gg_grading(df):

    for index, row in df.iterrows():

        grader_group_id = df[df['student_id'] == row['grader_id']]['group_id'].unique()

        if len(grader_group_id) == 0:
            df.at[index, 'grader_group_id'] = 0
            continue

        df.at[index, 'grader_group_id'] = grader_group_id[0]


    df_array = []
    for index, row in df.iterrows():

        row = {
            'group_id': row['group_id'],
            'task_id': str(row['sheet'])+str(row['task']),
            'grade': row['grade'],
            'grader_group_id': row['grader_group_id'],
            'grader_id': row['grader_id'],
            'grader_type': row['grader_type']
        }

        if row in df_array:
            continue

        df_array.append(row)


    df = pd.DataFrame(df_array)

    return df


def make_gg(df_group):

    groups = df_group.groupby(['group_id', 'task_id', 'grader_group_id', 'grader_type'])
    df_array = []
    count_3 = 0
    for name, group in groups:

        if len(group['grader_id'].unique()) == 3:
            count_3 = count_3 + 1

        row = {
                'group_id': name[0],
                'task_id': name[1],
                'grader_group_id': name[2],
                'grader_type': name[3],
                'grade': round(group['grade'].mean(), 2)
        }

        df_array.append(row)

    df_group_group = pd.DataFrame(df_array)

    df_group_group.to_csv(REAL_DATA_PROCESSED_DIR + 'group-group.csv', index=False)

    return df_group_group


def run():
    print("--- Create group group setup for real data ----\n")

    print("1/4 -> create_temp")
    temp_df = create_temp()

    print("2/4 -> add_ta_grade")
    temp_df = add_ta_grade(temp_df)

    print("3/4 -> make_gg_grading")
    temp_df = make_gg_grading(temp_df)

    print("4/4 -> make_gg")
    gg = make_gg(temp_df)

    return gg

