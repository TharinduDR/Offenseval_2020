def filter_supportfile(df):
    df['subtask_a'] = df["average"].apply(lambda x: label(x))
    filtered_df = df.loc[(df['std'] < 0.1)]
    return filtered_df


def label(x):
    if x > 0.5:
        return 'OFF'
    else:
        return 'NOT'