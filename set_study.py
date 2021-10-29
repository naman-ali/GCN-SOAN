import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--study', default='number_of_graders', type=str, help='Run the desired study')
args = parser.parse_args()
study = args.study

studies = [
    'number_of_graders',
    'graders_bias',
    'ground_truth_distribution',
    'working_impact_grading',
    'erdos',
    'homophily',
    'real_data_peer_evaluation',
    'real_data_peer_and_self_evaluation',
    'custom'
]

if study not in studies:
    print("please selest a valid study identifier")
    sys.exit()

# study = "custom"

# unique identifiers that you can use to reproduce studies in paper:
# 1 -> 'number_of_graders',
# 2 -> 'graders_bias',
# 3 -> 'ground_truth_distribution',
# 4 -> 'working_impact_grading',
# 5 -> 'erdos',
# 6 -> 'homophily'
# 7 -> 'real_data_peer_evaluation'
# 8 -> 'real_data_peer_and_self_evaluation'
# 9 -> 'custom'


custom_study = {
    'class_size': 500,
    'number_of_graders': 3,
    'left_peak_mean': 0.3,  # if you choose normal distribution left_peak_mean will be ignored.
    'left_peak_std': 0.1,  # if you choose normal distribution left_peak_std will be ignored.
    'right_peak_mean': 0.7,
    'right_peak_std': 0.1,
    'distribution': 'bidmoal',  # possible values: bimodal | normal
    'graders_std': 0.25,
    'graders_bias': 0.0
}

Fast_TunedModel = True