def filter_supportfile(df, threshold):
    df['subtask_a'] = df["average"].apply(lambda x: label(x))
    filtered_df = df.loc[(df['std'] < threshold)]
    return filtered_df


def label(x):
    if x > 0.5:
        return 'OFF'
    else:
        return 'NOT'
