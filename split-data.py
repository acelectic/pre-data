import random

random.seed(432)

train_ratio = .85

full_data = open('pigeon.csv', 'r').readlines()
full_size = len(full_data)

print(full_data)
random.shuffle(full_data)

train = full_data[:int(full_size*train_ratio)]
test = full_data[int(full_size*train_ratio):]

print('train:{}\ttest:{}'.format(len(train), len(test)))

header = 'image,xmin,ymin,xmax,ymax,label\n'

with open('train-{}.csv'.format(train_ratio), 'w') as f:
    f.write(header+''.join( train))

with open('test-{}.csv'.format(train_ratio), 'w') as f:
    f.write(header+''.join(test))