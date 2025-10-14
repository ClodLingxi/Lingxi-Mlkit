import pandas as pd
import numpy as np

def _padding_np_array(target_list: np.ndarray, max_len, copy_padding=False, padding_direction='left', padding_value=0):
    if copy_padding:
        return np.array(max_len * [target_list])

    len_diff = max_len - len(target_list)
    if len_diff <= 0:
        return target_list

    if padding_direction == 'left':
        target_list = np.pad(target_list, (len_diff, 0), mode='constant', constant_values=padding_value)
    else:
        target_list =  np.pad(target_list, (0, len_diff), mode='constant', constant_values=padding_value)
    return target_list


def group_by_and_padding(
        pd_data: pd.DataFrame, group_target, return_mask=True, return_np=True, padding_direction='left', padding_value=np.nan
):
    pd_data = pd_data.groupby(group_target).agg(list).reset_index()
    other_columns_list = list(set(pd_data.columns) - {group_target})
    for column in other_columns_list:
        pd_data[column] = pd_data[column].apply(np.array)

    max_len = pd_data[other_columns_list[0]].apply(len).max()
    for column in other_columns_list:
        pd_data[column] = pd_data[column].apply(
            lambda x: _padding_np_array(x, max_len, padding_direction=padding_direction, padding_value=padding_value)
        )
    pd_data[group_target] = pd_data[group_target].apply(
        lambda x: _padding_np_array(x, max_len, copy_padding=True)
    )


    if return_np:
        result_np = np.array(pd_data.to_numpy().tolist()).swapaxes(-1, -2)
        if return_mask:
            if np.isnan(padding_value):
                mask = ~np.isnan(result_np[:, :, -1])
            else:
                mask = result_np[:, :, -1] == padding_value
            mask = mask[..., np.newaxis]
            result_np = np.concatenate([result_np, mask], axis=-1)
        return result_np

    if return_mask:
        if np.isnan(padding_value):
            pd_data["MASK"] = pd_data[other_columns_list[0]].apply(lambda x: ~np.isnan(x))
        else:
            pd_data["MASK"] = pd_data[other_columns_list[0]].apply(lambda x: x != padding_value)

    return pd_data


def apply_with_mask(target_data, mask, func):
    target_data = target_data[mask]
    target_data = func(target_data)
    return target_data
