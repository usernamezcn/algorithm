import pandas as pd
import numpy as np
import datetime
import random

df = pd.DataFrame(columns=['city_id','region_id','datetime','infected_num'])
# infection = pd.read_csv('../train_data/city_'+'A'+'/infection.csv',low_memory=False, infer_datetime_format=True,header=None,names = ['city_id','region_id','datetime','infected_num'])
# va = []
# k = infection['infected_num'][:45].values
# va.append(range(1,46))
# va.append(k)
# print(va[1])
# data = pd.DataFrame(k)
# data.to_csv('./data.csv',sep=',',index=False,header=False)

def lstmm(raw_seq):
    not_zero = 0
    for i in raw_seq:
        if i!=0:not_zero+=1
    if not_zero>30:not_zero = 30
    if (not_zero is np.nan) or (not_zero==0) :not_zero= 1000
    re_seq = []
    sum = 0.0
    k_2 = [0.3,0.15,0.075,0.05,0.02,0.01]
    k_3 = [0.15,0.05,0.0,0.0,0.0,0.0]
    k = np.arange(0,1,1/60)
    for i in range(60):
        sum+=(raw_seq[i]*k[i])
    sum = int(sum/not_zero)
    for i in range(0,6):
        beg, end = int(k_2[i] * sum), int(k_3[i] * sum)
        for j in range(5):
            re_seq.append(random.randint(end,beg))
    # print(re_seq)

    return re_seq



for city in ['A','B','C','D','E']:

    infection = pd.read_csv('../../B_train_data/train_data_all/city_'+city+'/infection.csv',low_memory=False, infer_datetime_format=True,header=None,names = ['city_id','region_id','datetime','infected_num'])
    infection_num = infection['infected_num']
    region = infection.drop_duplicates('region_id')
    length = len(region)
    print(length)
    for i in range(length):
        ward = infection_num[i * 60:i * 60 + 60].values.tolist()
        append_to = lstmm(ward)
        # print(type(append_to))
        now = datetime.datetime(2120,6,30)
        for j in range(30):
            # print(len(append_to))
            time = (now+datetime.timedelta(j)).strftime('%Y%m%d')
            # df2 = pd.DataFrame(np.transpose['A',i,time,0],columns=['city_id','region_id','datetime','infected_num'])
            app = {'city_id':[city],'region_id':[i],'datetime':[time],'infected_num':[append_to[j]]}
            app = pd.DataFrame(app)
            # print("app:",app)
            # print(type(app))
            df = df.append(app,ignore_index=True)



for city in ['F','G','H','I','J','K']:

    infection = pd.read_csv('../../B_train_data/train_data_all/city_'+city+'/infection.csv',low_memory=False, infer_datetime_format=True,header=None,names = ['city_id','region_id','datetime','infected_num'])

    region = infection.drop_duplicates('region_id')
    length = len(region)
    print(length)
    for i in range(length):
        now = datetime.datetime(2120,6,30)
        for j in range(30):
            time = (now+datetime.timedelta(j)).strftime('%Y%m%d')
            # df2 = pd.DataFrame(np.transpose['A',i,time,0],columns=['city_id','region_id','datetime','infected_num'])
            if j<10:
                app = {'city_id':[city],'region_id':[i],'datetime':[time],'infected_num':[50-2*j]}
            elif j>=10 and j<25:
                app = {'city_id': [city], 'region_id': [i], 'datetime': [time], 'infected_num': [30 - 2 * (j-10)]}
            else:
                app = {'city_id': [city], 'region_id': [i], 'datetime': [time], 'infected_num': [2]}
            app = pd.DataFrame(app)
            # print(app)
            df = df.append(app,ignore_index=True)








print(len(df))

df.to_csv('./submission.csv',index = False,header = False)