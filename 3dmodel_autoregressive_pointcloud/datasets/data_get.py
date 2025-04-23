import os
import numpy as np
cmd_root = '/home/rl4citygen/gsh/data/CMDGen/Sketch_1_Extrude_1_test/cmd'
cad_root = '/home/rl4citygen/gsh/data/CMDGen/Sketch_1_Extrude_1_test/cad'

save_path = '/home/rl4citygen/gsh/data/CMDGen'
cmd_names = os .listdir(cmd_root)
cmd_names.sort()
nums  = []
for name in cmd_names:
    nums.append(name[:-3])
#print(nums)
pic_root = '/home/rl4citygen/gsh/data/CMDGen/Sketch_1_Extrude_1_test/pic'

file_pathes = []
for num in nums:
    cad_data_path = os.path.join(cad_root, num+'.npy')
    cmd_data_path = os.path.join(cmd_root, num+'.h5')
    # pic_path_f = pic_root+'/'+f'abc_{num[:4]}_step_v00'+'/'+num+'/'+num+'_f.png'
    # pic_path_r = pic_root+'/'+f'abc_{num[:4]}_step_v00'+'/'+num+'/'+num+'_r.png'
    # pic_path_t = pic_root+'/'+f'abc_{num[:4]}_step_v00'+'/'+num+'/'+num+'_t.png'
    pic_path_f = pic_root + '/'+num+'_f.png'
    pic_path_r = pic_root + '/'+num+'_r.png'
    pic_path_t = pic_root + '/'+num+'_t.png'

    if  os.path.exists(cad_data_path) and os.path.exists(pic_path_f)\
         and os.path.exists(pic_path_r) and os.path.exists(pic_path_t) and os.path.exists(cmd_data_path):
        file_pathes.append(num)
        with open(os.path.join(save_path, 'all_data.txt'),'a') as f:
            f.write(num+'\n')
    else:
        print('num: ',num)
