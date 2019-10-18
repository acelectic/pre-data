
a = open('data/img/train-0.85.csv', 'r').readlines()
print(len(a))

b = open('data/img/test-0.85.csv', 'r').readlines()
print(len(b))


for i in range(100):
    print(i, end='\r')