
import numpy as np
import glob
import h5py
root = '/scratch/sg7484/data/CMDGen/Sketch_1_Extrude_1/cmd'
save_root = '/scratch/sg7484/data/CMDGen/Sketch_1_Extrude_1/cmd_txt'
all_file_path = glob.iglob(root + '/*.h5')
for file in all_file_path:
    # # print(file)
    # name = file.split('/')[-1]
    # name = name.split('.')[0]
    # print('file: ',file)
    # print(name)
    # cad_vec = np.load(file,allow_pickle=True)
    # save_path = save_root + '/' + name + '.txt'
    # np.savetxt(file, cad_vec)
    with h5py.File(file,"r") as f:
        print(f['vec'][:])
        
[  4  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1]
[  0 211 128  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1]
[  1 211 153 128   1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1]
[  0 128 153  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1]
[  1 128 128 128   1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1]
[  4  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1]
[  2 128 140  -1  -1   3  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1]
[  4  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1]
[  2 211 140  -1  -1   2  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1]
[  5  -1  -1  -1  -1  -1 128 128 128 128 115 128  96 132 128   0   0]
[  3  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1]


4  -1  -1  -1  
0 211 128  -1
1 211 153 128
0 128 153  -1
1 128 128 128
4  -1  -1  -1 
2 128 140  -1
4  -1  -1  -1
2 211 140  -1 




