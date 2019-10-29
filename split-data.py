import random

random.seed(432)

train_ratio = .85

suffix = 'nikko'
file_pigeon ='pigeon_{}.csv'.format(suffix)
full_data = open(file_pigeon, 'r').readlines()
full_size = len(full_data)

print(full_data)
random.shuffle(full_data)

train = full_data[:int(full_size*train_ratio)]
test = full_data[int(full_size*train_ratio):]

print('train:{}\ttest:{}'.format(len(train), len(test)))

header = 'image,xmin,ymin,xmax,ymax,label\n'
header=''

with open('train-{suffix}-{train_ratio}.csv'.format(suffix=suffix, train_ratio=train_ratio), 'w') as f:
    f.write(header+''.join( train))

with open('test-{suffix}-{train_ratio}.csv'.format(suffix=suffix, train_ratio=train_ratio), 'w') as f:
    f.write(header+''.join(test))