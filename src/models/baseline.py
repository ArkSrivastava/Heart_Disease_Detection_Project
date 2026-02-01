def baseline_rule(df):
    th=df['chol'].mean()
    return (df['chol']>th).astype(int)
