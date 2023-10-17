'''
Cars dataset downloaded from https://www.kaggle.com/datasets/iamsouravbanerjee/cars-dataset/

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')


from worker import *


def main():
    parser = argparse.ArgumentParser(description='Cars dataset')
    parser.add_argument('-d', '--data_path', type=str, required=True, help='Path to cars dataset')
    parser.add_argument('-o', '--output_path', type=str, required=True, help='Path to output directory')

    args = parser.parse_args()
    data_path = args.data_path

    # if output directory does not exist, create it
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # read data
    df = pd.read_csv(data_path)

    # check for missing values
    check_missing_values(df)

    # check for duplicate rows
    check_duplicate_rows(df)

    # check for outliers
    check_outliers(df)

    # check for class imbalance
    check_class_imbalance(df)

    # check for correlation between features
    check_correlation(df)

    # check for distribution of features
    check_distribution(df)

    # check for distribution of target
    check_target_distribution(df)

    # check for distribution of features with respect to target
    check_feature_target_distribution(df)



if __name__ == '__main__':
    main()