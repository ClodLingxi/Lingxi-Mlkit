from typing import Sequence

import numpy as np
import pandas as pd


def handle_numpy_data(table_data: np.ndarray, discrete_flag: Sequence[bool], return_unique_dict=False):
    assert table_data.ndim == 2
    r_dict = {}

    for column_idx in range(table_data.shape[1]):
        flag = discrete_flag[column_idx]
        if not flag:
            continue

        np_unique = np.unique(table_data[:, column_idx], return_inverse=True)
        table_data[:, column_idx] = np_unique[1]

        if return_unique_dict:
            r_dict[column_idx] = np_unique[0]

    if not return_unique_dict:
        return table_data
    else:
        return table_data, r_dict


def auto_detect_discrete_column(table_data: np.ndarray, threshold: float = 0) -> np.ndarray:
    assert table_data.ndim == 2, "Input must be a 2D array"
    n_rows, n_cols = table_data.shape
    discrete_flags = []

    for column_idx in range(n_cols):
        column = table_data[:, column_idx]

        is_string_column = False

        if column.dtype.kind in ['U', 'S']:
            is_string_column = True
        elif column.dtype == object or column.dtype.kind == 'O':
            sample_value = None
            for val in column:
                if val is not None and val != '':
                    sample_value = val
                    break

            if sample_value is not None:
                try:
                    float(sample_value)
                    is_string_column = False
                except (ValueError, TypeError):
                    is_string_column = True

        if is_string_column:
            discrete_flags.append(len(set(column)))
        else:
            unique_count = len(np.unique(column))
            unique_ratio = unique_count / n_rows
            is_discrete_column = unique_ratio <= threshold
            discrete_flags.append(unique_count if is_discrete_column else False)

    return np.array(discrete_flags)


if __name__ == '__main__':
    x = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [4, 5, 6],
        "c": ["a", "b", "c"],
    }).to_numpy()

    discrete_column = auto_detect_discrete_column(x)

    print(discrete_column)

    result = handle_numpy_data(x.copy(), discrete_column)

    print(result)
