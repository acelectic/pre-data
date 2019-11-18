
# a = open('data/img/train-0.85.csv', 'r').readlines()
# print(len(a))

# b = open('data/img/test-0.85.csv', 'r').readlines()
# print(len(b))


# for i in range(100):
#     print(i, end='\r')
import numpy as np
import pprint
# from glob import glob

# a = glob('data4eval/non-pigeon/*.jpg')
# print(a)
# b = glob('data4eval/non-pigeon/*.png')
# print(b)

# pprint.pprint(a+b)

a = np.array([a for a in range(10)])
a = np.cumsum(a) / 10.
pprint.pprint(a)