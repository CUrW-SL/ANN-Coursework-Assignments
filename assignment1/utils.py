import pandas as pd
from assignment1 import DATA_SET_FP, attribute_formation, attribute_converters


def read_data_set():
    df = pd.read_csv(DATA_SET_FP, sep=',', index_col=False, header=None, names=attribute_formation)
    # remove rows with missing values.
    # 1. mark rows with missing values False
    _missing_values_marked_series = (df[attribute_formation] != '?').all(axis=1)
    # 2. get the rows marked as True
    df = df[_missing_values_marked_series]

    # convert column values accordingly
    df = df.transform(attribute_converters)[attribute_formation]
    return df


if __name__ == '__main__':
    print(read_data_set())
