import glob, os
from shutil import copyfile

# test_file = 'test_0.85-333.csv'
test_file = 'test_merge.csv'

test_temp = None

with open(test_file) as f:
    test_temp = f.readlines()
    f.close()

print(len(test_temp))

base_dir = os.getcwd()
new_test_dir = base_dir+'/data4eval/test_merge'
# groundtruths_dir = base_dir+'/data4eval/groundtruths'
print(base_dir)

os.makedirs(new_test_dir, exist_ok=True)

tmp_truths = {}

with open('data4eval/{}'.format(test_file), 'w') as f:
    for i in test_temp[:]:
        
        tmp_split = i.split(',')
        print('tmp_splt', tmp_split)
        name_img = tmp_split[0].replace('"', '')
        old_img = base_dir + '/' + name_img
        new_name_img = new_test_dir + '/' + name_img.split('/')[-1]
     
        print(old_img, new_name_img)   
        copyfile(old_img, new_name_img)
        

        new_i = ','.join([new_name_img.replace(base_dir+'/data4eval/','')]+tmp_split[1:])
        print(repr(new_i))
        
        f.write(new_i)
        # tr_name = name_img.split('/')[-1]
        # try:
        #     tmp_truths[tr_name] += [{
        #         'label':tmp_split[-1],
        #         'box': (tmp_split[1], tmp_split[2], tmp_split[3], tmp_split[4])
        #     }]
        # except:
        #     tmp_truths[tr_name] = [{
        #         'label':tmp_split[-1],
        #         'box': (tmp_split[1], tmp_split[2], tmp_split[3], tmp_split[4])
        #     }]

    # print(tmp_truths)

# os.makedirs(groundtruths_dir, exist_ok=True)

# for key, data in tmp_truths.items():
#     with open(groundtruths_dir+ '/' + key.replace('.png', '.txt'), 'w') as f:
#         for data_2 in data:
#             f.write(data_2['label'].replace('\n', '').replace('"', '') + ' ' + ' '.join(data_2['box']) + '\n')
    