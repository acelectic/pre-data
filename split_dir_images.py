import glob, os
from shutil import copyfile

test_file = 'test_0.85-333.csv'
test_temp = None

with open(test_file) as f:
    test_temp = f.readlines()
    f.close()

print(len(test_temp))

base_dir = os.getcwd()
new_test_dir = base_dir+'/data4eval/test'
print(base_dir)

os.makedirs(new_test_dir, exist_ok=True)

header = 'image,xmin,ymin,xmax,ymax,label\n'

with open('data4eval/{}'.format(test_file), 'w') as f:
    f.write(header)
    for i in test_temp[:]:
        tmp_split = i.split(',')
        print(tmp_split)
        name_img = tmp_split[0].replace('"', '')
        old_img = base_dir + '/' + name_img
        new_name_img = new_test_dir + '/' + name_img.split('/')[-1]
        
        copyfile(old_img, new_name_img)
        print(old_img, new_name_img)

        new_i = ','.join([tmp_split[0].replace('data/img/', 'data4eval/test/' )]+tmp_split[1:])
        print(repr(new_i))
        f.write(new_i)

    