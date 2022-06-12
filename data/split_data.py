import pandas as pd
import numpy as np

train = pd.read_csv('raw/patent/train_original.csv', index_col='id')
test = pd.read_csv('raw/patent/test_original.csv', index_col='id')

anchor_list = list(train['anchor'].unique())
anchor_list.append(list(test['anchor'].unique()))

test['score'] = np.zeros(test.shape[0])
full = train.append(test)
print(full)

anchor_dict = {}
a = 0
for i in range(full.shape[0]):
    if {full.iloc[i, 0]: full.iloc[i, 2]} not in anchor_dict.values():
        anchor_dict[a] = {full.iloc[i, 0]: full.iloc[i, 2]}
        a += 1

id_list = []
for i in range(full.shape[0]):
    val = {full.iloc[i, 0]: full.iloc[i, 2]}
    for k, v in anchor_dict.items():
        if v == val:
            id_list.append(k)
full['id_anchor'] = id_list
print(full)

file_a = pd.DataFrame()
file_a['id'] = full['id_anchor']
file_a['anchor'] = full['anchor']
file_a['context'] = full['context']
file_a.set_index('id', inplace=True)
file_a.drop_duplicates(inplace=True)
file_a.reset_index(inplace=True)

target_dict = {}
a = 0
for i in range(full.shape[0]):
    if {full.iloc[i, 1]: full.iloc[i, 2]} not in target_dict.values():
        target_dict[a] = {full.iloc[i, 1]: full.iloc[i, 2]}
        a += 1

id_list = []
for i in range(full.shape[0]):
    val = {full.iloc[i, 1]: full.iloc[i, 2]}
    for k, v in target_dict.items():
        if v == val:
            id_list.append(k)
full['id_target'] = id_list
print(full)

file_b = pd.DataFrame()
file_b['id'] = full['id_target']
file_b['target'] = full['target']
file_b['context'] = full['context']
file_b.set_index('id', inplace=True)
file_b.drop_duplicates(inplace=True)
file_b.reset_index(inplace=True)

'''
file_a['id'] = pd.concat([train['id'], test['id']])
file_a['anchor'] = pd.concat([train['anchor'], test['anchor']])
file_a['context'] = pd.concat([train['context'], test['context']])

file_b = pd.DataFrame()
file_b['id'] = pd.concat([train['id'], test['id']])
file_b['target'] = pd.concat([train['target'], test['target']])
file_b['context'] = pd.concat([train['context'], test['context']])
'''

train_file = pd.DataFrame()
train_file['ltable_id'] = full['id_anchor'][:int(0.8*train.shape[0])]
train_file['rtable_id'] = full['id_target'][:int(0.8*train.shape[0])]
train_file['label'] = full['score'][:int(0.8*train.shape[0])]

valid_file = pd.DataFrame()
valid_file['ltable_id'] = full['id_anchor'][int(0.8*train.shape[0]):train.shape[0]]
valid_file['rtable_id'] = full['id_target'][int(0.8*train.shape[0]):train.shape[0]]
valid_file['label'] = full['score'][int(0.8*train.shape[0]):train.shape[0]]

test_file = pd.DataFrame()
test_file['ltable_id'] = full['id_anchor'][train.shape[0]:]
test_file['rtable_id'] = full['id_target'][train.shape[0]:]
test_file['label'] = np.zeros(test.shape[0])

file_a.set_index('id', inplace=True)
file_a.to_csv('raw/patent/tableA.csv')
file_b.set_index('id', inplace=True)
file_b.to_csv('raw/patent/tableB.csv')
train_file.set_index('ltable_id', inplace=True)
train_file.to_csv('raw/patent/train.csv')
valid_file.set_index('ltable_id', inplace=True)
valid_file.to_csv('raw/patent/valid.csv')
test_file.set_index('ltable_id', inplace=True)
test_file.to_csv('raw/patent/test.csv')