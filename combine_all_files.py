
import shutil
import os

# script tested

root_path = '/home/workspace/CarND-Behavioral-Cloning-P3/'
   
stats=[]
source_dirs = []
source_dirs.append(root_path+'data_custom_2_laps_succ/IMG/')        # 2 laps (18.05.2021)
source_dirs.append(root_path+'data_custom_obr/IMG/')    # 1 lap
source_dirs.append(root_path+'data_custom_vozvr/IMG/')  # 1 lap
##source_dirs.append(root_path+'data_custom_track2/IMG/') # 1 lap

print(source_dirs)

target_dir = root_path+'data_custom_ALL/IMG/'
    
# loo through all source dirs
# loo through all source dirs

for source_dir in source_dirs:
          
    print(source_dir)
    # get list of all files in source dir
    file_names = os.listdir(source_dir)
    # cope all files to destination folder
    for i,file_name in enumerate(file_names):
        shutil.copy(os.path.join(source_dir, file_name), target_dir)
       
    stats.append(source_dir+" : " + str(i+1))
    
      
file_names = os.listdir(target_dir)
    # cope all files to destination folder
    
k = 0
for file  in file_names:
    k+=1

print(target_dir+" : "+str(k))
print(stats)
