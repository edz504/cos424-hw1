from sklearn.metrics import roc_curve
import pandas as pd
import numpy as np

probs_df = pd.read_csv('pred_prob.csv')

full_roc_df = pd.DataFrame(columns=['tpr', 'fpr', 'model'])
for col in probs_df.columns[1:]:
    fpr, tpr, thresh = roc_curve(probs_df['truth'], probs_df[col])
    roc_df = pd.DataFrame({
        'tpr': tpr,
        'fpr': fpr,
        'model' : np.repeat(col, len(tpr))})
    full_roc_df = pd.concat([full_roc_df, roc_df])
    
full_roc_df.to_csv('roc_custom.csv', index=False)
