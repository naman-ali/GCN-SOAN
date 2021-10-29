from requires.helper import *


class Group:


    def __init__(self, df, group_id):

        self.df = df
        self.group_id = int(group_id)


    def get_grades(self, grader=None, task_id=None, grader_type=None):

        df_student = self.df[self.df['group_id'] == self.group_id]

        if task_id is not None:
            df_student = df_student[df_student['task_id'] == task_id]

        if grader is not None:
            df_student = df_student[df_student['grader_group_id'] == grader]

        if grader_type is not None:
            df_student = df_student[df_student['grader_type'] == grader_type]


        return df_student


    def get_gradings(self, task_id=None, student=None):

        df_student = self.df[self.df['grader_group_id'] == self.group_id]

        if task_id is not None:
            df_student = df_student[df_student['task_id'] == task_id]

        if student is not None:
            df_student = df_student[df_student['group_id'] == student]

        return df_student


    def get_graders(self, task_id=None):

        df_student = self.df[self.df['group_id'] == self.group_id]

        if task_id is not None:
            df_student = df_student[df_student['task_id'] == task_id]

        return df_student['grader_group_id'].unique().tolist()


    def get_grade_by_grader(self, grader):

        return self.get_grades(grader=grader)['grade'].iloc[0]

    def get_gradings_groups(self):

        return self.get_gradings()['group_id'].unique().tolist()