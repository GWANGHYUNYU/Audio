import pickle
import os

import numpy as np
import matplotlib.pylab as plt

path = 'D:\Code_Torch\GE\pickle_data\Vu'
filename = '_Generation_Bool_Arr.pickle'

i = 0

totalname = path + '\\' + str(i) + filename

# # load
# with open(totalname, 'rb') as f:
#     pair_data = pickle.load(f)
#
# acc, bool_arr = zip(*pair_data)
#
# max_point = np.argmax(acc)
# print(np.max(acc), max_point)
# print('%s Generation \t Max ACCURACY: %s \n BOOL ARRAY: %s'%(i, acc[max_point], bool_arr[max_point]))
#
# bool_sum = np.sum(bool_arr[max_point])
# print(bool_sum)
#
# plt.figure(figsize=(18, 5))
# plt.bar(np.arange(1, 176), bool_arr[max_point], width=0.5) # bar chart 만들기
# plt.xticks(np.arange(1, 176), rotation=90, fontsize=9)
#
# plt.title('★Accuracy: ' + str(acc[max_point]) + '     ★Trainable Layers: '+ str(bool_sum))
# plt.xlabel('Layers')
# plt.ylabel('Trainable')
# plt.show()
# # plt.tight_layout()
# # plt.savefig('savefig_default_18.png')

# for i in enumerate():

folders = os.listdir(path)
sort_folders = sorted(folders)

# for idx, file in enumerate(folders):
#     print(idx)
#     print(file)

path = 'D:\Code_Torch\GE\pickle_data\Vu'
filename = '_Generation_Bool_Arr.pickle'

for i in range(151):
    totalname = path + '\\' + str(i) + filename
    print(totalname)
    save_filename = path + '\\array\\' + str(i) + filename[:-7]
    # load
    with open(totalname, 'rb') as f:
        pair_data = pickle.load(f)

    acc, bool_arr = zip(*pair_data)

    max_point = np.argmax(acc)
    print(np.max(acc), max_point)
    print('%s Generation \t Max ACCURACY: %s \n BOOL ARRAY: %s'%(i, acc[max_point], bool_arr[max_point]))

    bool_sum = np.sum(bool_arr[max_point])
    print(bool_sum)

    plt.figure(figsize=(18, 5))
    plt.bar(np.arange(1, 176), bool_arr[max_point], width=0.5) # bar chart 만들기
    plt.xticks(np.arange(1, 176), rotation=90, fontsize=9)

    plt.title('★Accuracy: ' + str(acc[max_point]) + '     ★Trainable Layers: '+ str(bool_sum))
    plt.xlabel('Layers')
    plt.ylabel('Trainable')
    # plt.show()
    plt.tight_layout()
    plt.savefig(save_filename + '.png')