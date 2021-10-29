from requires.config import *
import pandas as pd


class PeerRank:
    # constant attributes
    d = 0.7
    iterations = 20

    def __init__(self, experiment_id, df):

        self.experiment_id = experiment_id
        self.df = self.get_df()
        self.class_size = len(self.df)
        self.number_of_graders = len(self.df.iloc[0]['graders'].split(','))

        peer_ranks = self.run_model()
        self.save()

    def get_df(self):
        experiment_file = STUDY_DIR + str(self.experiment_id) + '.csv'
        df = pd.read_csv(experiment_file)
        return df


    def run_model(self):

        # create a zero numpy array with class_size, class_size dimension
        G = np.zeros(shape=(self.class_size, self.class_size)).astype(float)

        for student_id, student in self.df.iterrows():
            G[student_id, student_id] = 0
            graders = student['graders'].split(',')
            grades = student['grades'].split(',')
            grades_dict = dict(zip(graders, grades))

            for grader, grade in grades_dict.items():
                G[student_id, int(grader)] = grade

        G = np.transpose(G)
        row_count, col_count = G.shape
        G_copy = G.astype(float).copy()
        G_transpose = np.transpose(G_copy)

        # print(np.unique(G))
        # sys.exit()

        # average grade as initial ranks
        r = []
        for each in G_transpose:
            r.append(each.sum() / self.number_of_graders)
        previous_r = r.copy()

        for iteration in range(1, self.iterations):
            print("PeerRank iteration:", iteration, "/", self.iterations, "grades samples:", r[1], r[10], r[20])
            for node_i in range(0, row_count):

                sigma = 0
                sum_graders_grades = 0

                for inlink in self.inlinks(node_i, G_copy):
                    sigma = sigma + previous_r[inlink] * G_transpose[node_i, inlink]
                    sum_graders_grades = sum_graders_grades + previous_r[inlink]

                if sum_graders_grades != 0:
                    r[node_i] = (1 - self.d) * previous_r[node_i] + (self.d / sum_graders_grades) * sigma
                    r[node_i] = round(r[node_i], 20)
                else:
                    r[node_i] = 0

            previous_r = r

        for index, each in enumerate(r):
            self.df.at[index, 'PeerRank'] = each

        return self.df

    def save(self):

        experiment_file = STUDY_DIR + str(self.experiment_id) + '.csv'
        return self.df.to_csv(experiment_file, index=False)


    @staticmethod
    def outlinks(node, G):
        row_count, col_count = G.shape
        outlinks = []
        for col in range(0, col_count):
            if G[node, col] != 0:
                outlinks.append(col)

        return outlinks

    @staticmethod
    def inlinks(node, G):
        row_count, col_count = G.shape
        G = np.transpose(G)
        inlinks = []
        for col in range(0, col_count):
            if G[node, col] != 0:
                inlinks.append(col)

        return inlinks

