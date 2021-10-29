#!/usr/bin/python
# Copyright Luca de Alfaro, 2013.

import numpy as np
import pandas as pd
from requires.config import ROOT_DIR

REAL_DATA_PROCESSED_DIR = ROOT_DIR + 'data/real_data_processed/'

# Do we debias grades?
DEBIAS = True
# Aggregation using median?  Generally a bad idea.
AGGREGATE_BY_MEDIAN = False
# Basic precision, as multiple of standard deviation.
BASIC_PRECISION = 0.1
# Uses also data from a vertex in order to send a message to that vertex?
USE_ALL_DATA = True


class User:

    def __init__(self, name):
        """Initializes a user."""
        self.name = name
        self.items = set()
        self.grade = {}

    def add_item(self, it, grade):
        self.items = self.items | set([it])
        self.grade[it] = grade


class Item:

    def __init__(self, id):
        self.id = id
        self.users = set()
        self.grade = None

    def add_user(self, u):
        self.users = self.users | set([u])


class Graph:

    def __init__(self, basic_precision=None, use_all_data=USE_ALL_DATA):

        self.items = set()
        self.users = set()
        self.user_dict = {}
        self.item_dict = {}
        self.basic_precision = basic_precision
        self.use_all_data = use_all_data

    def add_review(self, username, item_id, grade):
        # Gets, or creates, the user.
        if username in self.user_dict:
            u = self.user_dict[username]
        else:
            u = User(username)
            self.user_dict[username] = u
            self.users = self.users | set([u])
        # Gets, or creates, the item.
        if item_id in self.item_dict:
            it = self.item_dict[item_id]
        else:
            it = Item(item_id)
            self.item_dict[item_id] = it
            self.items = self.items | set([it])
        # Adds the connection between the two.
        u.add_item(it, grade)
        it.add_user(u)

    def get_user(self, username):
        return self.user_dict.get(username)

    def get_item(self, item_id):
        return self.item_dict.get(item_id)

    def evaluate_items(self, n_iterations=20, precision=None):
        """Evaluates items using the reputation system iterations."""
        # Builds the initial messages from users to items.
        for it in self.items:
            it.msgs = []
            for u in it.users:
                m = Msg()
                m.user = u
                m.grade = u.grade[it]
                m.variance = 1.0
                it.msgs.append(m)
        # Does the propagation iterations.
        for i in range(n_iterations):
            self._propagate_from_items()
            self._propagate_from_users()
        # Does the final aggregation step.
        self._aggregate_item_messages()
        self._aggregate_user_messages()

    # Evaluates each item via average voting.
    def avg_evaluate_items(self):
        item_value = {}
        for it in self.items:
            grades = []
            for u in it.users:
                grades.append(u.grade[it])
            it.grade = aggregate(grades)

    def _propagate_from_items(self):
        """Propagates the information from items to users."""
        # First, clears all incoming messages.
        for u in self.users:
            u.msgs = []
        # For each item, gives feedback to the users.
        for it in self.items:
            # For each user that evaluated the item, reports to that user the following
            # quantities, computed from other users:
            # Average/median
            # Standard deviation
            # Total weight
            for u in it.users:
                if len(it.msgs) > 0:
                    grades = []
                    variances = []
                    for m in it.msgs:
                        if self.use_all_data or m.user != u or len(it.msgs) < 2:
                            grades.append(m.grade)
                            variances.append(m.variance)
                    variances = np.array(variances)
                    weights = 1.0 / (self.basic_precision + variances)
                    weights /= np.sum(weights)
                    msg = Msg()
                    msg.item = it
                    msg.grade = aggregate(grades, weights=weights)
                    # Now I need to estimate the standard deviation of the grade.
                    msg.variance = np.sum(variances * weights * weights)
                    u.msgs.append(msg)

    def _propagate_from_users(self):
        """Propagates the information from users to items."""
        # First, clears the messages received in the items.
        for it in self.items:
            it.msgs = []
        # Sends information from users to items.
        # The information to be sent is a grade, and an estimated standard deviation.
        for u in self.users:
            for it in u.items:
                if len(u.msgs) > 0:
                    # The user looks at the messages from other items, and computes
                    # what has been the bias of its evaluation.
                    msg = Msg()
                    msg.user = u
                    biases = []
                    weights = []
                    if DEBIAS:
                        for m in u.msgs:
                            if self.use_all_data or m.item != it or len(u.msgs) < 2:
                                weights.append(1 / (self.basic_precision + m.variance))
                                given_grade = u.grade[m.item]
                                other_grade = m.grade
                                biases.append(given_grade - other_grade)
                        u.bias = aggregate(biases, weights=weights)
                    else:
                        u.bias = 0.0
                    # The grade is the grade given, de-biased.
                    msg.grade = u.grade[it] - u.bias
                    # Estimates the standard deviation of the user, from the
                    # other judged items.
                    variance_estimates = []
                    weights = []
                    for m in u.msgs:
                        if self.use_all_data or m.item != it or len(u.msgs) < 2:
                            it_grade = u.grade[m.item] - u.bias
                            variance_estimates.append((it_grade - m.grade) ** 2.0)
                            weights.append(1.0 / (self.basic_precision + m.variance))
                    msg.variance = aggregate(variance_estimates, weights=weights)
                    # The message is ready for enqueuing.
                    it.msgs.append(msg)

    def _aggregate_item_messages(self):
        """Aggregates the information on an item, computing the grade
        and the variance of the grade."""
        for it in self.items:
            it.grade = None
            it.variance = None
            if len(it.msgs) > 0:
                grades = []
                variances = []
                for m in it.msgs:
                    grades.append(m.grade)
                    variances.append(m.variance)
                variances = np.array(variances)
                weights = 1.0 / (self.basic_precision + variances)
                weights /= np.sum(weights)
                it.grade = aggregate(grades, weights=weights)
                it.variance = np.sum(variances * weights * weights)

    def _aggregate_user_messages(self):
        """Aggregates the information on a user, computing the
        variance and bias of a user."""
        for u in self.users:
            u.variance = None
            if len(u.msgs) > 0:
                biases = []
                weights = []
                # Estimates the bias.
                if DEBIAS:
                    for m in u.msgs:
                        weights.append(1 / (self.basic_precision + m.variance))
                        given_grade = u.grade[m.item]
                        other_grade = m.grade
                        biases.append(given_grade - other_grade)
                    u.bias = aggregate(biases, weights=weights)
                else:
                    u.bias = 0.0
                # Estimates the grade for each item.
                variance_estimates = []
                weights = []
                for m in u.msgs:
                    it_grade = u.grade[m.item] - u.bias
                    variance_estimates.append((it_grade - m.item.grade) ** 2.0)
                    weights.append(1.0 / (self.basic_precision + m.variance))
                u.variance = aggregate(variance_estimates, weights=weights)

    def evaluate_users(self):
        """Evaluates users by comparing their variance with the one computed by
        giving grades at random."""
        # Computes the standard deviation of all grades ever given.
        all_grades = []
        for u in self.users:
            all_grades.extend(u.grade.values())
        overall_stdev = np.std(all_grades)
        # The stdev of the difference between two numbers is sqrt(2) times the
        # stdev of the numbers, assuming normal distribution.
        overall_stdev *= 2 ** 0.5
        for u in self.users:
            u.quality = max(0.0, 1.0 - (u.variance ** 0.5) / overall_stdev)

    def avg_evaluate_users(self):
        """Evaluates users from item grades, as if no messages had actually been sent."""
        # Computes the standard deviation of all grades ever given.
        all_grades = []
        for u in self.users:
            all_grades.extend(u.grade.values())
        overall_stdev = np.std(all_grades)
        # The stdev of the difference between two numbers is sqrt(2) times the
        # stdev of the numbers, assuming normal distribution.
        overall_stdev *= 2 ** 0.5
        for u in self.users:
            diffs = []
            for it in u.items:
                d = it.grade - u.grade[it]
                diffs.append(d * d)
            u.variance = np.average(diffs)
            u.quality = max(0.0, 1.0 - (u.variance ** 0.5) / overall_stdev)


class Msg():
    def __init__(self):
        pass


def aggregate(v, weights=None):
    """Aggregates using either average or median."""
    if AGGREGATE_BY_MEDIAN:
        if weights is not None:
            return median_aggregate(v, weights=weights)
        else:
            return median_aggregate(v)
    else:
        if weights is not None:
            return np.average(v, weights=weights)
        else:
            return np.average(v)


def median_aggregate(values, weights=None):
    if len(values) == 1:
        return values[0]
    if weights is None:
        weights = np.ones(len(values))
    # Sorts.
    vv = []
    for i in range(len(values)):
        if weights[i] > 0:
            vv.append((values[i], weights[i]))
    if len(vv) == 0:
        return values[0]
    if len(vv) == 1:
        x, _ = vv[0]
        return x
    vv.sort()
    v = np.array([x for x, _ in vv])
    w = np.array([y for _, y in vv])
    # print 'v', v, 'w', w
    # At this point, the values are sorted, they all have non-zero weight,
    # and there are at least two values.
    half = np.sum(w) / 2.0
    below = 0.0
    i = 0
    while i < len(v) and below + w[i] < half:
        below += w[i]
        i += 1
    # print 'i', i, 'half', half, 'below', below
    if half < below + 0.5 * w[i]:
        # print 'below'
        if i == 0:
            return v[0]
        else:
            alpha = half - below
            beta = below + 0.5 * w[i] - half
            # print 'alpha', alpha, 'beta', beta
            return (beta * (v[i] + v[i - 1]) / 2.0 + alpha * v[i]) / (alpha + beta)
    else:
        # print 'above'
        if i == len(v) - 1:
            # print 'last'
            return v[i]
        else:
            alpha = half - below - 0.5 * w[i]
            beta = below + w[i] - half
            return (beta * v[i] + alpha * (v[i] + v[i + 1]) / 2.0) / (alpha + beta)


def save(g, task_id=None):

    models_df = pd.read_csv(REAL_DATA_PROCESSED_DIR + 'gg-models-results.csv')

    for item in g.items:
        row_index = models_df[(models_df['group_id'] == item.id) & (models_df['task_id'] == task_id)].index
        models_df.at[row_index, 'Vancouver_grade'] = item.grade

    models_df.to_csv(REAL_DATA_PROCESSED_DIR + "gg-models-results.csv", index=False)

    return True


def run(df=None, precision=0.1):

    g = Graph(basic_precision=precision)

    for index, row in df.iterrows():
        g.add_review(row['grader_group_id'], row['group_id'], row['grade'])

    g.evaluate_items(precision=precision)

    save(g, task_id=df['task_id'].iloc[0])
