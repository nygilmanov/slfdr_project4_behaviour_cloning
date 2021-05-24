
import shutil
import os


import preprocess as pp


root_path = '/home/workspace/CarND-Behavioral-Cloning-P3/'


final=[]
   
sample_2laps  =  pp.get_samples(root_path,'data_custom_2_laps_succ/driving_log.csv')
sample_obr    =  pp.get_samples(root_path,'data_custom_obr/driving_log.csv')
sample_vozvr  =  pp.get_samples(root_path,'data_custom_vozvr/driving_log.csv')
sample_track2 =  pp.get_samples(root_path,'data_custom_track2/driving_log.csv')


print('sample_2laps : ',3*len(sample_2laps))
print('sample_obr : '  ,3*len(sample_obr))
print('sample_vozvr : '  ,3*len(sample_vozvr))
print('sample_track2 : '  ,3*len(sample_track2))

final=sample_2laps+sample_obr+sample_vozvr+sample_track2

print('total : '  ,3*len(final))


print(sample_2laps[0])

