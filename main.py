import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
# from xgboost import XGBClassifier
from model import EXP
import tensorflow as tf

with open('./features.txt', 'r') as f:
    data = f.read()
    tmps = data.split('\n')
    data_new = [item.split(':')[0] for item in tmps]
    details = [item.split(':')[-1] for item in tmps]
CSV_HEADER = data_new
train_name = list(set(data_new)-set(['salary', 'marital_stat']))
# print(train_name)
print(data_new.index('salary'))
print(data_new.index('marital_stat'))

train_data_file = ("dataset_income/census/census-income.data")
train_data = pd.read_csv(train_data_file, header=None, names=CSV_HEADER)

test_data_file = ("dataset_income/census/census-income.test")
test_data = pd.read_csv(test_data_file, header=None, names=CSV_HEADER)

vali_data_file = ("dataset_income/census/census-income.vali")
vali_data = pd.read_csv(vali_data_file, header=None, names=CSV_HEADER)

train_label1 = 1*(train_data['salary'] == ' 50000+.')
test_label1 = 1*(test_data['salary'] == ' 50000+.')
vali_label1 = 1*(vali_data['salary'] == ' 50000+.')

train_label2 = 1*(train_data['marital_stat'] != ' Never married')
test_label2 = 1*(test_data['marital_stat'] != ' Never married')
vali_label2 = 1*(vali_data['marital_stat'] != ' Never married')

print(train_label1.mean())
print(train_label2.mean())

print(f"Train dataset shape: {train_data.shape}")
print(f"Test dataset shape: {test_data.shape}")
# print(train_data.head())

train = train_data.values
for i, detail in enumerate(details):
    if detail == ' continuous.':
        continue
    else:
        key = data_new[i]
        for j, item in enumerate(detail[:-1].split(',')):
            if j == 0:
                tmp = j * (train_data[key] == item)
            else:
                tmp += j * (train_data[key] == item)
        train[:, i] = tmp

test = test_data.values
for i, detail in enumerate(details):
    if detail == ' continuous.':
        # test[:, i] = test_data[key].astype(np.float64)
        continue
    else:
        key = data_new[i]
        for j, item in enumerate(detail[:-1].split(',')):
            if j == 0:
                tmp = j * (test_data[key] == item)
            else:
                tmp += j * (test_data[key] == item)
        test[:, i] = tmp

vali = vali_data.values
for i, detail in enumerate(details):
    if detail == ' continuous.':
        # test[:, i] = test_data[key].astype(np.float64)
        continue
    else:
        key = data_new[i]
        for j, item in enumerate(detail[:-1].split(',')):
            if j == 0:
                tmp = j * (vali_data[key] == item)
            else:
                tmp += j * (vali_data[key] == item)
        vali[:, i] = tmp

train_sparse = [[[] for _ in range(229285)] for _ in range(32)]
train_dense = [[] for _ in range(229285)]
count_train = 0
for i, detail in enumerate(details):
    key = data_new[i]
    if key in ['salary', 'marital_stat']:
        continue
    if detail == ' continuous.':
        for j in range(len(train)):
            train_dense[j] += [train[j][i]]
    else:
        alls = detail[:-1].split(',')
        for j in range(len(train)):
            tmp = [0]*len(alls)
            tmp[train[j][i]] = 1
            train_sparse[count_train][j] += tmp
        # print(count_train)
        count_train += 1

# test = test_data.values
test_sparse = [[[] for _ in range(70000)] for _ in range(32)]
test_dense = [[] for _ in range(70000)]
count_test = 0
for i, detail in enumerate(details):
    key = data_new[i]
    if key in ['salary', 'marital_stat']:
        continue
    if detail == ' continuous.':
        for j in range(len(test)):
            test_dense[j] += [test[j][i]]
    else:
        alls = detail[:-1].split(',')
        for j in range(len(test)):
            tmp = [0]*len(alls)
            tmp[test[j][i]] = 1
            test_sparse[count_test][j] += tmp
        count_test += 1

vali_sparse = [[[] for _ in range(20000)] for _ in range(32)]
vali_dense = [[] for _ in range(20000)]
count_test = 0
for i, detail in enumerate(details):
    key = data_new[i]
    if key in ['salary', 'marital_stat']:
        continue
    if detail == ' continuous.':
        for j in range(len(vali)):
            vali_dense[j] += [vali[j][i]]
    else:
        alls = detail[:-1].split(',')
        for j in range(len(vali)):
            tmp = [0]*len(alls)
            tmp[vali[j][i]] = 1
            vali_sparse[count_test][j] += tmp
        count_test += 1


test_dense = np.array(test_dense)
train_dense = np.array(train_dense)
vali_dense = np.array(vali_dense)

def train_input_fn():
    global BATCH
    print('batch_size is: ', BATCH)
    train_datas, train_label = {}, {}
    for i in range(32):
        name = 'sparse%d'%i
        train_datas[name] = tf.convert_to_tensor(np.array(train_sparse[i]).astype(np.float64))
    train_datas['dense'] = tf.convert_to_tensor(train_dense.astype(np.float64))

    train_datas['salaryWeight'] = tf.ones_like(train_label1)
    train_datas['marital_statWeight'] = tf.ones_like(train_label2)
    train_label['salary'] = tf.convert_to_tensor(train_label1)
    train_label['marital_stat'] = tf.convert_to_tensor(train_label2)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_datas, train_label))
    # train_labels = tf.data.Dataset.from_tensor_slices(train_label)
    return train_dataset.shuffle(buffer_size=229285).batch(BATCH)#, train_labels

# a = train_input_fn()
# iterator = a.make_one_shot_iterator()
# next_element = iterator.get_next()
# sess = tf.Session()
# print(sess.run(next_element))

def test():
    test_datas, test_label = {}, {}
    # for i, name in enumerate(train_name):
    #   test_datas[name] = test_features[:, i].astype(np.float64)
    # test_datas['salary'] = test_label1.astype(np.int64)
    # test_datas['marital_stat'] = test_label2.astype(np.int64)
    # test_dataset = tf.data.Dataset.from_tensor_slices(test_datas)
    for i in range(32):
        name = 'sparse%d'%i
        test_datas[name] = tf.convert_to_tensor(np.array(test_sparse[i]).astype(np.float64))
    test_datas['dense'] = tf.convert_to_tensor(test_dense.astype(np.float64))
    # weight = np.ones_like(test_label1)
    # shape = weight.shape
    # test_datas['salaryWeight'] = tf.convert_to_tensor(np.where((test_label1 <= 0) & (np.random.uniform(low=0, high=1, size=shape) >= 0.2), np.zeros_like(weight), weight))
    test_datas['salaryWeight'] = tf.ones_like(test_label1)
    test_datas['marital_statWeight'] = tf.ones_like(test_label2)
    test_label['salary'] = tf.convert_to_tensor(test_label1)
    test_label['marital_stat'] = tf.convert_to_tensor(test_label2)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_datas, test_label))
    # train_labels = tf.data.Dataset.from_tensor_slices(train_label)
    return test_dataset.batch(1)

def vali():
    vali_datas, vali_label = {}, {}
    # for i, name in enumerate(train_name):
    #   test_datas[name] = test_features[:, i].astype(np.float64)
    # test_datas['salary'] = test_label1.astype(np.int64)
    # test_datas['marital_stat'] = test_label2.astype(np.int64)
    # test_dataset = tf.data.Dataset.from_tensor_slices(test_datas)
    for i in range(32):
        name = 'sparse%d'%i
        vali_datas[name] = tf.convert_to_tensor(np.array(vali_sparse[i]).astype(np.float64))
    vali_datas['dense'] = tf.convert_to_tensor(vali_dense.astype(np.float64))
    # weight = np.ones_like(test_label1)
    # shape = weight.shape
    # test_datas['salaryWeight'] = tf.convert_to_tensor(np.where((test_label1 <= 0) & (np.random.uniform(low=0, high=1, size=shape) >= 0.2), np.zeros_like(weight), weight))
    vali_datas['salaryWeight'] = tf.ones_like(vali_label1)
    vali_datas['marital_statWeight'] = tf.ones_like(vali_label2)
    vali_label['salary'] = tf.convert_to_tensor(vali_label1)
    vali_label['marital_stat'] = tf.convert_to_tensor(vali_label2)
    vali_dataset = tf.data.Dataset.from_tensor_slices((vali_datas, vali_label))
    # train_labels = tf.data.Dataset.from_tensor_slices(train_label)
    return vali_dataset.batch(32)

import global_var
global_var._init()
results = {}
global BATCH
global learning_rate
global optimizer
global max_iter
# max_iter = 20
# BATCH = 32
# global_var.set_value('optimizer', 'adam')
# global_var.set_value('learning_rate', 0.001)

# with open('./result.txt', 'a+') as f:
#     my_estimator = tf.estimator.Estimator(model_fn=EXP)
#     for iter in range(max_iter):
#         # my_estimator = tf.estimator.Estimator(model_fn=MMOE)
#         my_estimator.train(input_fn=train_input_fn, steps=500000)
#     res = my_estimator.evaluate(input_fn=test)
#     print(res)

# trian
run_config = tf.estimator.RunConfig(
    model_dir='./mymodel/ckpt1/',
    save_checkpoints_steps=1000,
    keep_checkpoint_max=5)

with open('./temp.txt', 'a+') as f:
    for optimizer in ['adam']:# , 'adgrad', 'sgd']:
        for max_iter in [16]:
            for BATCH in [32]:
                for learning_rate in [0.0005]:
                    global_var.set_value('optimizer', optimizer)
                    global_var.set_value('learning_rate', learning_rate)
                    # my_estimator = tf.estimator.Estimator(model_fn=MMOE)
                    my_estimator = tf.estimator.Estimator(model_fn=EXP, config=run_config)
                    # early_stopping = tf.estimator.experimental.stop_if_no_increase_hook(my_estimator, metric_name='auc/salary', max_steps_without_increase=4, min_steps=128800, run_every_steps=1, run_every_secs=None)
                    # train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, hooks=[early_stopping])
                    # eval_spec = tf.estimator.EvalSpec(input_fn=vali)
                    for iter in range(max_iter):
                        my_estimator.train(input_fn=train_input_fn, steps=500000)
                        # tf.estimator.train_and_evaluate(my_estimator, train_spec, eval_spec)
                    res = my_estimator.evaluate(input_fn=test)
                    results[optimizer+'_'+str(max_iter)+'_'+str(BATCH)+'_'+str(learning_rate)] = res
                    print(results)
                    f.write(optimizer+'_'+str(max_iter)+'_'+str(BATCH)+'_'+str(learning_rate)+': ')
                    f.write('\t')
                    f.write(str(res))
                    f.write('\n')

# # test
# run_config = tf.estimator.RunConfig(
#     model_dir='/home/jovyan/mycode/mymodel/demo/ckpt1/',
#     save_checkpoints_steps=1000,
#     keep_checkpoint_max=5)
# my_estimator = tf.estimator.Estimator(model_fn=EXP, config=run_config)
# res = my_estimator.evaluate(input_fn=test)
# print(res)


# print(results)