import glob, os
import cv2
import random, time

random.seed(432)
print(cv2.__version__)

name_class = {}
for i, e in enumerate(open('data/obj.names','r').read().split('\n')):
    name_class[i] = e
print(name_class)

data_dir = os.getcwd()+ '/data/img_nikko/'

def to_path(name):
    return 'data/img_nikko/'+name

def clean_bbox(bbox, image_name, h, w):
    tmp = [x.replace('\n', '') for x in bbox]
    tmp = [x.split(' ') for x in tmp]
    tmp = [list(map(float, x)) for x in tmp]
    # print('\tbbox2 {}'.format(repr(tmp)))
    # print('\t', h,' ', w)
    result = []
    for i in tmp:
        name_tmp = name_class[int(i[0])]
        x_cen = i[1]*w
        y_cen = i[2]*h
        w_ = i[3]*w
        h_ = i[4]*h

        x1 = int(abs(x_cen - (w_/2)))
        y1 = int(abs(y_cen - (h_ / 2)))
        x2 = int(abs(x_cen + (w_ / 2)))
        y2 = int(abs(y_cen + (h_ / 2)))

        # x1 = abs(x_cen - (w_ / 2))
        # y1 = abs(y_cen - (h_ / 2))
        # x2 = abs(x_cen + (w_ / 2))
        # y2 = abs(y_cen + (h_ / 2))

        # print('{}\t{}\t{}\t{}\t{}'.format(name_tmp, x_cen, y_cen, w_, h_))
        # print('{}\t{}\t{}\t{}\t{}'.format(name_tmp, x1, y1, x2, y2))
        result.append([image_name, x1, y1, x2, y2, name_tmp])
        # result += ['{},{},{},{},{},{}'.format(image_name, x1, y1, x2, y2, name_tmp)]
    return result




full_data = []
nf_datas = {}

rtext = glob.glob('data/img_nikko/*.txt')
print(len(rtext))

btext = glob.glob('data/img/*.txt')

print(len(btext))

ttext = rtext+btext
rtext = ttext

with open('final_train.csv', 'w') as f_pigeon:
    
    tmp_i = 1
    l_ = len(rtext)
    print(l_)
    print('START\n')
    for txt_name in rtext:
        th_='YOLO_2_CSV:'+ str(tmp_i) + '----' + str(l_)
        # print(th_, end='\r')
        # print('sss', end='\r')
        tmp_i += 1
        tmp = txt_name.split('/')[-1].replace('.txt', '')
        # print(tmp)
        data_dir = '/'.join(txt_name.split('/')[:-1]) + '/'
        # print(data_dir)
        
        # print(image_name, )
        bbox = open(txt_name, 'r').readlines()
        # print(bbox)
        if len(bbox) > 0:
            
            image_namet = data_dir + tmp + '.jpg'
            # print(image_namet)
            image_name = tmp + '.jpg'
            if os.path.isfile(image_namet):
                # print('file jpg')
                pass
            else:
                # print('file png')
                
                image_namet = data_dir + tmp + '.png'
                # print(image_namet)
                image_name = tmp + '.png'
            
            img = cv2.imread(image_namet)
            h, w = img.shape[0], img.shape[1]
            # print('do it',txt_name, '\t', image_name)
            # print('\tbbox {}'.format(repr(bbox)))
            result = clean_bbox(bbox, image_name, h=h, w=w)
            # print('\tresult{}'.format(result))

            for v, i in enumerate(result):
                if i[5] == 'pigeon':
                    full_data += [i]
                    t_ = '"{}",{},{},{},{},"{}"\n'.format(to_path(i[0]), i[1], i[2], i[3], i[4], i[5])
                    # print(t_)
                    f_pigeon.write(t_)
                    
                    try:
                        nf_datas[to_path(i[0])] += [[i[1], i[2], i[3], i[4], i[5]]]
                    except:
                        nf_datas[to_path(i[0])] = [[i[1], i[2], i[3], i[4], i[5]]]

                # elif i[5] == 'kajok':
                #     t_ = '"{}",{},{},{},{},"{}"\n'.format(to_path(i[0]), i[1], i[2], i[3], i[4], i[5])
                #     # print(t_)
                #     f_kajok.write(t_)
            #     cv2.rectangle(img, (i[1], i[2]), (i[3], i[4]), (0,255,0), 3)
            #
            # cv2.imshow('ss', img)
            # cv2.waitKey()
        else:
            # print(txt_name)
            pass
        
        
# for i, boxs in nf_datas.items():
#     print(i)
#     print(boxs)







print(len(nf_datas.keys()))
d_keys = list(nf_datas.keys())
random.shuffle(d_keys)

train_ratio = .85
full_size = len(d_keys)

train_keys = d_keys[:int(full_size*train_ratio)]
test_keys = d_keys[int(full_size*train_ratio):]
# train = full_data[:int(full_size*train_ratio)]
# test = full_data[int(full_size*train_ratio):]


train = []
for i in train_keys:
    for data in nf_datas[i]:
        train += ['{key},{data}'.format(key=i, data=','.join(map(str, data)))]

tests = []
for i in test_keys:
    for data in nf_datas[i]:
        tests += ['{key},{data}'.format(key=i, data=','.join(map(str, data)))]

print('trainKeys {}\ntestKeys {}'.format(len(train_keys), len(test_keys)))
print('trainAnnos {}\ntestAnnos {}'.format(len(train), len(tests)))


# print(bg_list)
# header = 'image,xmin,ymin,xmax,ymax,label\n'

with open('train-split_{}-{}-({}).csv'.format(train_ratio, len(train_keys), len(train)), 'w') as f:
    for i in train:
        f.write('{}\n'.format(i))
    
with open('test-split_{}-{}-({}).csv'.format(train_ratio, len(test_keys), len(tests)), 'w') as f:
    for i in tests:
        f.write('{}\n'.format(i))





print('NEG')
bg_list = []
negs = glob.glob('data/newnegs/*/*')

for i in negs:
    if not ("pigoen" in i or "Pigeon" in i):
        bg_list += [','.join(map(str, [i,'','','','','']))]

negs = glob.glob('data/img/neg*')

for i in negs:
    bg_list += [','.join(map(str, [i,'','','','','']))]

neg_ratio_of_train = int(len(train)/.85*.15)
print(neg_ratio_of_train)
print('train:{}\ntest:{}\nneg:{}'.format(len(train), len(tests), neg_ratio_of_train))


print(len(bg_list))
random.shuffle(bg_list)

train_neg = train + bg_list[:neg_ratio_of_train]
random.shuffle(train_neg)
print("train-neg: {}".format(len(train_neg)))
with open('train-final-neg_{}-{}-({})-{}.csv'.format(train_ratio, len(train_keys), len(train), neg_ratio_of_train), 'w') as f:
    for i in train_neg:
        f.write('{}\n'.format(i))

with open('all_neg_{}.csv'.format(len(bg_list)), 'w') as f:
    for i in bg_list[:]:
        f.write('{}\n'.format(i))
        
print('copy train test')
with open('train-merge.csv', 'w') as f:
    for i in train_neg:
        f.write('{}\n'.format(i))
    
with open('test-merge.csv', 'w') as f:
    for i in tests:
        f.write('{}\n'.format(i))