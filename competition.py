# Method Description:
# I use model based method. At the beginning, I tried to improve the performance of HW3,
# which is hybrid method with both memory-based and model-based. But with the improvement of
# the model-based part, memory-based method cannot enhance the result anymore. So I just focus on
# the model.
# I select stars, review_count, categories, hours from business file, time from checkin file.
# average_stars,review_count, yelping_since, useful, friends, funny, cool, fans, elite and every
# compliment_* attr from user file, count tips number for every
# user and business from tip file, photo number for business from photo
# file, useful, funny, cool from review_train file. Then handle these attr by calculating average,
# total, length, most frequent, etc.. Finally, I use XGBRegressor to train the model, modify params like
# random_state, n_estimators, max_depth, learning_rate to get the best model

# Error Distribution:
# >=0 and <1: 102227
# >=1 and <2: 32820
# >=2 and <3: 6165
# >=3 and <4: 830
# >=4: 2

# RMSE: 0.9795695758154569

# Execution Time: 315.1058340072632s


import sys
from pyspark import SparkContext
import time
import json
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import datetime

def merge_dict(d1, d2):
    d1.update(d2)
    return d1

def category_num(cat_str):
    if cat_str==None:
        return 0
    length=len(cat_str.split(','))
    return length

def time_num(hours):
    if hours==None:
        return 0
    return len(hours)

def transit_cat(all_cat):
    if scat in all_cat:
        return 1
    else:
        return 0



# folder = sys.argv[1]
# test_file = sys.argv[2]
# output_file = sys.argv[3]
folder = 'HW3StudentData'
test_file = 'HW3StudentData/yelp_val.csv'
output_file = 'out.csv'

t0=time.time()
sc=SparkContext('local[*]', 'competition')
sc.setLogLevel('WARN')

train_data=sc.textFile(folder+'/yelp_train.csv').filter(lambda x: x.startswith('user_id') == False)
test_data=sc.textFile(test_file).filter(lambda x: x.startswith('user_id') == False)
train_user=train_data.map(lambda x: (x.split(',')[0], 1)).reduceByKey(lambda x, y: x)\
    .map(lambda x: (1, [x[0]])).reduceByKey(lambda x, y: x+y).first()[1]
train_bus=train_data.map(lambda x: (x.split(',')[1], 1)).reduceByKey(lambda x, y: x)\
    .map(lambda x: (1, [x[0]])).reduceByKey(lambda x, y: x+y).first()[1]
test_user=test_data.map(lambda x: (x.split(',')[0], 1)).reduceByKey(lambda x, y: x)\
    .map(lambda x: (1, [x[0]])).reduceByKey(lambda x, y: x+y).first()[1]
test_bus=test_data.map(lambda x: (x.split(',')[1], 1)).reduceByKey(lambda x, y: x)\
    .map(lambda x: (1, [x[0]])).reduceByKey(lambda x, y: x+y).first()[1]
all_user=list(set(train_user+test_user))
all_bus=list(set(train_bus+test_bus))
user_code=dict(zip(all_user, range(len(all_user))))
code_to_user=dict(zip(range(len(all_user)), all_user))
bus_code=dict(zip(all_bus, range(len(all_bus))))
code_to_bus=dict(zip(range(len(all_bus)), all_bus))
user_basket=train_data.map(lambda x: (user_code[x.split(',')[0]], (bus_code[x.split(',')[1]], float(x.split(',')[2]))))\
    .groupByKey().map(lambda x: (1, {x[0]: dict(x[1])})).reduceByKey(lambda x, y: merge_dict(x, y))\
    .map(lambda x: x[1]).collect()[0]
bus_basket=train_data.map(lambda x: (bus_code[x.split(',')[1]], (user_code[x.split(',')[0]], float(x.split(',')[2]))))\
    .groupByKey().map(lambda x: (1, {x[0]: dict(x[1])})).reduceByKey(lambda x, y: merge_dict(x, y))\
    .map(lambda x: x[1]).collect()[0]
for u in user_code.values():
    if u not in user_basket.keys():
        user_basket[u]={}
for b in bus_code.values():
    if b not in bus_basket.keys():
        bus_basket[b]={}

# 要改
bus_tempp=sc.textFile(folder+'/business.json').map(lambda x: json.loads(x))
b_data=bus_tempp\
    .map(lambda x: (x['business_id'], x['stars'], x['review_count'], category_num(x['categories']), time_num(x['hours']))).collect()
check_data=sc.textFile(folder+'/checkin.json').map(lambda x: json.loads(x))\
    .map(lambda x: (x['business_id'], x['time']))\
    .map(lambda x: (x[0], sum(list(x[1].values())), sum(list(x[1].values()))/len(list(x[1].values())))).collect()
u_data=sc.textFile(folder+'/user.json').map(lambda x: json.loads(x))\
    .map(lambda x: (x['user_id'], x['average_stars'], x['review_count'],
                    (datetime.date(2022, 4, 1)-datetime.date(int(x['yelping_since'].split('-')[0]),
                                                            int(x['yelping_since'].split('-')[1]),
                                                            int(x['yelping_since'].split('-')[2]))).days,
                    x['useful'], category_num(x['friends']), x['funny'], x['cool'], x['fans'],
                    category_num(x['elite']), x['compliment_hot'], x['compliment_more'], x['compliment_profile'],
                    x['compliment_cute'], x['compliment_cute'], x['compliment_note'], x['compliment_plain'],
                    x['compliment_cool'], x['compliment_funny'], x['compliment_writer'], x['compliment_photos']))\
    .collect()
tip_data_temp=sc.textFile(folder+'/tip.json').map(lambda x: json.loads(x))
tip_b=tip_data_temp.map(lambda x: (x['business_id'], 1)).reduceByKey(lambda x, y: x+y).collect()
tip_u=tip_data_temp.map(lambda x: (x['user_id'], 1)).reduceByKey(lambda x, y: x+y).collect()
photo_data=sc.textFile(folder+'/photo.json').map(lambda x: json.loads(x))\
    .map(lambda x: (x['business_id'], 1)).reduceByKey(lambda x, y: x+y).collect()
review_data_temp=sc.textFile(folder+'/review_train.json').map(lambda x: json.loads(x))
review_b=review_data_temp.map(lambda x: (x['business_id'], (x['useful'], x['funny'], x['cool'], 1)))\
    .reduceByKey(lambda x, y: (x[0]+y[0], x[1]+y[1], x[2]+y[2], x[3]+y[3]))\
    .map(lambda x: (x[0], x[1][0], x[1][0]/x[1][3], x[1][1], x[1][1]/x[1][3], x[1][2], x[1][2]/x[1][3])).collect()
review_u=review_data_temp.map(lambda x: (x['user_id'], (x['useful'], x['funny'], x['cool'], 1))) \
    .reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1], x[2] + y[2], x[3] + y[3])) \
    .map(lambda x: (x[0], x[1][0], x[1][0] / x[1][3], x[1][1], x[1][1] / x[1][3], x[1][2], x[1][2] / x[1][3])).collect()

bus_cat=bus_tempp.map(lambda x: (x['business_id'], x['categories'])).collect()
df_bus_cat=pd.DataFrame(bus_cat, columns=['business_id', 'categories'])
df_bus_cat['categories']=df_bus_cat['categories'].fillna('').apply(lambda x: x.split(', '))
aa=df_bus_cat['categories'].values
list_cat=[]
for eve in aa:
    list_cat.extend(eve)
set_cat=set(list_cat)
for scat in set_cat:
    df_bus_cat[scat]=df_bus_cat['categories'].apply(transit_cat)

top_num=8
cat_PCA=PCA(n_components=top_num).fit_transform(df_bus_cat.iloc[:, 2:])
cat_name=['cat1', 'cat2', 'cat3', 'cat4', 'cat5', 'cat6', 'cat7', 'cat8']
bus_final_cat=pd.DataFrame(cat_PCA, columns=cat_name)
bus_final_cat['business_id']=df_bus_cat['business_id']

b_dict={}
u_dict={}
# 要改
for b in b_data:
    b_dict[b[0]]=[b[1], b[2], b[3], b[4], 0, 0.0, 0, 0,  # 0-7
                  0, 0, 0, 0, 0, 0, # 8-13
                  0, 0, 0, 0, 0, # 14-18
                  0, 0, 0, 0, 0, 0]
                  # 0, 0, 0, 0, 0]
for c in check_data:
    b_dict[c[0]][4]=c[1]
    b_dict[c[0]][5]=c[2]
for p in photo_data:
    b_dict[p[0]][7]=p[1]
for tb in tip_b:
    b_dict[tb[0]][6] = tb[1]
for u in u_data:
    u_dict[u[0]] = [u[1], u[2], 0,
                    0, 0, 0, 0, 0, 0, u[3], u[4], u[5],
                    u[6], u[7], u[8], u[9],
                    u[10], u[11], u[12], u[13], u[14], u[15], u[16], u[17], u[18], u[19], u[20]]
for tu in tip_u:
    u_dict[tu[0]][2] = tu[1]
for rb in review_b:
    b_dict[rb[0]][8]=rb[1]
    b_dict[rb[0]][9]=rb[2]
    b_dict[rb[0]][10]=rb[3]
    b_dict[rb[0]][11]=rb[4]
    b_dict[rb[0]][12]=rb[5]
    b_dict[rb[0]][13]=rb[6]
for ru in review_u:
    u_dict[ru[0]][3]=ru[1]
    u_dict[ru[0]][4]=ru[2]
    u_dict[ru[0]][5]=ru[3]
    u_dict[ru[0]][6]=ru[4]
    u_dict[ru[0]][7]=ru[5]
    u_dict[ru[0]][8]=ru[6]
for idx, line in bus_final_cat.iterrows():
    b_dict[line['business_id']][14]=line['cat1']
    b_dict[line['business_id']][15]=line['cat2']
    b_dict[line['business_id']][16]=line['cat3']
    b_dict[line['business_id']][17]=line['cat4']
    b_dict[line['business_id']][18]=line['cat5']
    b_dict[line['business_id']][19]=line['cat6']
    b_dict[line['business_id']][20]=line['cat7']
    b_dict[line['business_id']][21]=line['cat8']
    # b_dict[line['business_id']][22]=line['cat9']
    # b_dict[line['business_id']][23]=line['cat10']
for uu in all_user:
    if uu not in u_dict.keys():
        u_dict[uu]=[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                    np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                    np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
for bb in all_bus:
    if bb not in b_dict.keys():
        b_dict[bb]=[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                    np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                    np.nan, np.nan, np.nan, np.nan, np.nan,
                    np.nan, np.nan, np.nan]

# 要改
train_tolist=train_data.map(lambda x: (x.split(',')[0], x.split(',')[1], x.split(',')[2]))\
    .map(lambda x: [user_code[x[0]], bus_code[x[1]], u_dict[x[0]][0], u_dict[x[0]][1], u_dict[x[0]][2],
                    u_dict[x[0]][3], u_dict[x[0]][4], u_dict[x[0]][5], u_dict[x[0]][6], u_dict[x[0]][7], u_dict[x[0]][8],
                    b_dict[x[1]][0], b_dict[x[1]][1], b_dict[x[1]][2], b_dict[x[1]][3], b_dict[x[1]][4],
                    b_dict[x[1]][5],b_dict[x[1]][6], b_dict[x[1]][7],
                    b_dict[x[1]][8], b_dict[x[1]][9], b_dict[x[1]][10], b_dict[x[1]][11], b_dict[x[1]][12], b_dict[x[1]][13],
                    b_dict[x[1]][14], b_dict[x[1]][15], b_dict[x[1]][16], b_dict[x[1]][17], b_dict[x[1]][18],
                    b_dict[x[1]][19], b_dict[x[1]][20], b_dict[x[1]][21], u_dict[x[0]][9], u_dict[x[0]][10],
                    u_dict[x[0]][11], u_dict[x[0]][12], u_dict[x[0]][13], u_dict[x[0]][14], u_dict[x[0]][15],
                    u_dict[x[0]][16], u_dict[x[0]][17], u_dict[x[0]][18], u_dict[x[0]][19], u_dict[x[0]][20],
                    u_dict[x[0]][21], u_dict[x[0]][22], u_dict[x[0]][23], u_dict[x[0]][24], u_dict[x[0]][25],
                    u_dict[x[0]][26],
                    float(x[2])])\
    .collect()
# 要改
df_train=pd.DataFrame(train_tolist, columns=['uid', 'bid', 'u_star', 'u_rate_num', 'u_tip',
                                             'u_useful_sum', 'u_useful_avg', 'u_funny_sum', 'u_funny_avg', 'u_cool_sum', 'u_cool_avg',
                                             'b_star',
                                             'b_rate_num', 'b_categories', 'b_time', 'b_checkin_sum',
                                             'b_checkin_avg', 'b_tip', 'b_photo',
                                             'b_useful_sum', 'b_useful_avg', 'b_funny_sum', 'b_funny_avg', 'b_cool_sum', 'b_cool_avg',
                                             'cat1', 'cat2', 'cat3', 'cat4', 'cat5',
                                             'cat6', 'cat7', 'cat8', 'u_since', 'u_useful', 'u_friend',
                                             'u_funny', 'u_cool', 'u_fans', 'u_elite',
                                             'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'c11',
                                             'star'])
df_train.set_index(['uid', 'bid'])
# 要改
test_tolist=test_data.map(lambda x: (x.split(',')[0], x.split(',')[1]))\
    .map(lambda x: [user_code[x[0]], bus_code[x[1]], u_dict[x[0]][0], u_dict[x[0]][1], u_dict[x[0]][2],
                    u_dict[x[0]][3], u_dict[x[0]][4], u_dict[x[0]][5], u_dict[x[0]][6], u_dict[x[0]][7], u_dict[x[0]][8],
                    b_dict[x[1]][0], b_dict[x[1]][1], b_dict[x[1]][2], b_dict[x[1]][3], b_dict[x[1]][4],
                    b_dict[x[1]][5], b_dict[x[1]][6], b_dict[x[1]][7],
                    b_dict[x[1]][8], b_dict[x[1]][9], b_dict[x[1]][10], b_dict[x[1]][11], b_dict[x[1]][12], b_dict[x[1]][13],
                    b_dict[x[1]][14], b_dict[x[1]][15], b_dict[x[1]][16], b_dict[x[1]][17], b_dict[x[1]][18],
                    b_dict[x[1]][19], b_dict[x[1]][20], b_dict[x[1]][21], u_dict[x[0]][9], u_dict[x[0]][10],
                    u_dict[x[0]][11], u_dict[x[0]][12], u_dict[x[0]][13], u_dict[x[0]][14], u_dict[x[0]][15],
                    u_dict[x[0]][16], u_dict[x[0]][17], u_dict[x[0]][18], u_dict[x[0]][19], u_dict[x[0]][20],
                    u_dict[x[0]][21], u_dict[x[0]][22], u_dict[x[0]][23], u_dict[x[0]][24], u_dict[x[0]][25],
                    u_dict[x[0]][26]])\
.collect()

# 要改
df_test=pd.DataFrame(test_tolist, columns=['uid', 'bid', 'u_star', 'u_rate_num', 'u_tip',
                                           'u_useful_sum', 'u_useful_avg', 'u_funny_sum', 'u_funny_avg', 'u_cool_sum', 'u_cool_avg',
                                           'b_star',
                                           'b_rate_num', 'b_categories', 'b_time', 'b_checkin_sum',
                                           'b_checkin_avg', 'b_tip', 'b_photo',
                                           'b_useful_sum', 'b_useful_avg', 'b_funny_sum', 'b_funny_avg', 'b_cool_sum', 'b_cool_avg',
                                           'cat1', 'cat2', 'cat3', 'cat4', 'cat5',
                                           'cat6', 'cat7', 'cat8', 'u_since', 'u_useful', 'u_friend',
                                           'u_funny', 'u_cool', 'u_fans', 'u_elite',
                                           'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'c11'])
df_test.set_index(['uid', 'bid'])
train_X=df_train[['uid', 'bid', 'u_star', 'u_rate_num', 'u_tip',
                  'u_useful_sum', 'u_useful_avg', 'u_funny_sum', 'u_funny_avg', 'u_cool_sum', 'u_cool_avg',
                  'b_star', 'b_rate_num', 'b_categories',
                  'b_time', 'b_checkin_sum', 'b_checkin_avg', 'b_tip', 'b_photo',
                  'b_useful_sum', 'b_useful_avg', 'b_funny_sum', 'b_funny_avg', 'b_cool_sum', 'b_cool_avg',
                  'cat1', 'cat2', 'cat3', 'cat4', 'cat5',
                  'cat6', 'cat7', 'cat8', 'u_since', 'u_useful', 'u_friend',
                  'u_funny', 'u_cool', 'u_fans', 'u_elite',
                  'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'c11']]
# xgbm=xgb.XGBRegressor(n_estimators=100, random_state=2)
# 要改
xgbm = xgb.XGBRegressor(random_state=2, n_estimators=250, max_depth=9, colsample_bytree=0.7,
                        learning_rate=0.05, subsample=1.0, min_child_weight=4,
                        reg_alpha=0.5, reg_lambda=0.5)

xgbm.fit(train_X, df_train['star'])
test_res2=xgbm.predict(df_test)

df_reset=df_test[['uid', 'bid']]
df_reset['mb_score']=test_res2
str_res='user_id, business_id, prediction\n'
for index, row in df_reset.iterrows():
    if row['mb_score']<1:
        row['mb_score']=1
    elif row['mb_score']>5:
        row['mb_score']=5
    str_res+=code_to_user[row['uid']]+','+code_to_bus[row['bid']]+','+str(row['mb_score'])+'\n'

with open(output_file, 'w') as f:
    f.write(str_res)

print(time.time()-t0)

rmse=0
temp_rmse=0
num_list=[0,0,0,0,0]
with open(output_file) as ff:
    predf=ff.readlines()[1:]
with open(test_file) as fff:
    realf=fff.readlines()[1:]
for i in range(len(predf)):
    temp_rmse=temp_rmse+(float(predf[i].split(',')[2])-float(realf[i].split(',')[2]))**2
    temp_diff=float(predf[i].split(',')[2])-float(realf[i].split(',')[2])
    if abs(temp_diff)<1:
        num_list[0]+=1
    elif 1<=abs(temp_diff)<2:
        num_list[1] += 1
    elif 2<=abs(temp_diff)<3:
        num_list[2] += 1
    elif 3<=abs(temp_diff)<4:
        num_list[3] += 1
    elif abs(temp_diff)>=4:
        num_list[4] += 1

rmse=(temp_rmse/len(predf))**0.5
print(rmse)

print('>=0 and <1: '+ str(num_list[0]))
print('>=1 and <2: '+ str(num_list[1]))
print('>=2 and <3: '+ str(num_list[2]))
print('>=3 and <4: '+ str(num_list[3]))
print('>=4: '+ str(num_list[4]))

