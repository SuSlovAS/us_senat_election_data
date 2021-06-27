import numpy as np
import pandas as pd
import scipy as sci

from matplotlib import pyplot as plt
import seaborn as sns

data_input = pd.read_csv('USSenate.csv')
data = data_input.copy()

#Research data
print(data.head(20))
print(data.tail(20))
print(data.describe())
print(data.isna().sum())
print(data.info())
print(data.shape)
print(data.corr())
print(data.columns.to_list())
print(data.items)
print('----'*20)
for col in data.columns.to_list():
    print(data[col].value_counts(ascending=True))
    print('----'*20)

print('----'*20)
data.rename(columns = {'unofficial_result':'unoff_res'},inplace=True)
data['unoff_res'] = data['unoff_res'].astype('int')
data['special election'] = data['special election'].astype('int')
data['write-in candidates'] = data['write-in candidates'].astype('int')
data['electoral stage'] = data['electoral stage'].apply(lambda x:1 if x=='gen' else 0)
def party_change(x):
    if x == 'LIBERTARIAN':
        return 0
    elif x == 'REPUBLICAN':
        return 1
    elif x == 'DEMOCRAT':
        return 2
    else:
        return 3
data['party_simplified'] = data['party_simplified'].apply(party_change)
data['candidate'].fillna('Unknown',inplace=True)

a = data['candidate_party'].value_counts(ascending=True)
def candidate_party(x):
    if 'INDEPENDENT' in x.upper():
        return 0
    elif 'LIBERTARIAN' in x.upper():
        return 1
    elif 'DEMOCRAT' in x.upper():
        return 2
    elif 'REPUBLICAN' in x.upper():
        return 3
    else:
        return 4
data['candidate_party'].fillna('NONE',inplace=True)
data['candidate_party'] = data['candidate_party'].apply(candidate_party)
data_cl = data.drop(['party_simplified','state','state_census_code','state_ICPSR'],axis=1)
corrPearson = data_cl.corr(method='pearson')
corrSpearman = data_cl.corr(method='spearman')
#Plot info
fig_1 = plt.figure(figsize=(10,8))
ax_1 = sns.heatmap(corrPearson,annot=True,vmin=-1,vmax=+1,
                   yticklabels=True,cbar=True,cmap='viridis')
sns.set_style('whitegrid')
plt.title('Pearson correlation')
plt.xlabel('Columns')
plt.ylabel('Columns')
plt.show()
fig_2 = plt.figure(figsize=(10,8))
ax_2 = sns.heatmap(corrSpearman,annot=True,vmin=-1,vmax=+1,
                   yticklabels=True,cbar=True,cmap='viridis')
sns.set_style('whitegrid')
plt.title('Spearman correlation')
plt.xlabel('Columns')
plt.ylabel('Columns')
plt.show()

fig_data = data_cl.hist(figsize = (20,10))
#Box plot
fig_3 = plt.figure(figsize=(10,8))
ax_3 = sns.boxplot(x='candidate votes',y='candidate_party',
                   data=data_cl[['candidate_party','candidate votes']],
                   orient='h',palette='Set3')
plt.show()
