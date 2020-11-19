

'''
Created on 2020. 10. 27.

@author: tobew
pip install xlrd 

'''
import pandas as pd
df=pd.read_excel('sample_data_1.xls',skiprows=0)


import numpy as np
# 길이를 200byte로 줄이기
datastream_val_array=df['Data'].values
print(datastream_val_array.shape)

new_datastream_list=[]
for one_line in datastream_val_array:
    one_line=one_line[:200]  # 200 바이트로 줄이기
    new_datastream_list.append(list(one_line))  # 문자열을 character의 array로 변경하기

print(np.array(new_datastream_list).shape)
print(len(new_datastream_list[0]))
print(new_datastream_list[0])
new_datastream_number_list=[]
for one_row in new_datastream_list:
    i=0
    new_one_list=[]
    for one_char in one_row:
        i+=1
        one_number=ord(one_char)  # 문자를 숫자로 바꾸기
        new_one_list.append(one_number)
    if i < 200:  # 200보다 적으면 0으로 채우기
        new_one_list +=[0.]*(200-i)
        
    new_datastream_number_list.append(new_one_list)

print(np.array(new_datastream_number_list).shape)
print(len(new_datastream_number_list[0]))
print(new_datastream_number_list[0])


#정규화
# 1바이트의 최대값은 255이므로 255로 나누어서 0~1사이 값으로 변경
new_normalized_number_list=[]
for one_row in new_datastream_number_list:
    new_one_row_list=[]
    for one_number in one_row:
        new_one_row_list.append(one_number/255.)
    new_normalized_number_list.append(new_one_row_list)

print(np.array(new_normalized_number_list).shape)
# (936, 200)
print(len(new_normalized_number_list[0]))
print(new_normalized_number_list[0])

# Attack 컬럼값과 DataStream을 변형한 값을 이용한 데이터셋 만들기   
attack_val_array=df['Attack'].values
#파일에 저장하기
new_dataset_list=list(zip(attack_val_array,new_normalized_number_list))
print(new_dataset_list[0])
# (0, [0.3137254901960784, 0.30980392156862746, 0.3254901960784

# file에 저장하기_total
with open('sample_total.csv','w',encoding='utf-8') as fout:
    for one_row in new_dataset_list:
        fout.write('{:}'.format(one_row[0]))
        for one_number in one_row[1]:
            fout.write(',{0:0.6f}'.format(one_number))
        fout.write('\n')
        
import random
# train, test 데이터셋 분할
f_train=open('train.csv','w',encoding='utf-8')
f_test=open('test.csv','w',encoding='utf-8')
ratio=0.9

for one_row in new_dataset_list:
    if random.random() > 0.1 :  # train data
        f_var=f_train
    else:
        f_var=f_test
    f_var.write('{:}'.format(one_row[0]))
    for one_number in one_row[1]:
        f_var.write(',{0:0.6f}'.format(one_number))
    f_var.write('\n')

