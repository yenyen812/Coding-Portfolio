from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
from scipy.stats import ttest_ind

# load the data
data_path = '/content/drive/MyDrive/111_result.csv'
df = pd.read_csv(data_path)

# compare consonants with 'th' or 'TH'
consonants = ['s', 'f', 'l', 't', 'd', 'n']
target_th = 'th'

# Position × Fake
conditions = [('onset', 'real'), ('onset', 'fake'), ('coda', 'real'), ('coda', 'fake')]

# Perform t-test
results = []

for pos, fake in conditions:
    print(f'\n--- Position={pos}, Type={fake} ---')
    for cons in consonants:
        group1 = df[(df['consonant'] == cons) & (df['Position'] == pos) & (df['Type'] == fake)]['COGlong']
        group2 = df[(df['consonant'] == target_th) & (df['Position'] == pos) & (df['Type'] == fake)]['COGlong']

        # check if all data got value
        if len(group1) > 1 and len(group2) > 1:
            t_stat, p_val = ttest_ind(group1, group2, equal_var=False)
            results.append({
                'Position': pos,
                'Type': fake,
                'Consonant': cons,
                't': t_stat,
                'p': p_val
            })
            print(f'{cons} vs {target_th}: t = {t_stat:.2f}, p = {p_val:.4f}')
        else:
            print(f'{cons} vs {target_th}: cannot perform test as lacking of data')

# Save the dataframe to Drive
ttest_df = pd.DataFrame(results)
ttest_df.to_csv('/content/drive/MyDrive/111_ttest_results.csv', index=False)