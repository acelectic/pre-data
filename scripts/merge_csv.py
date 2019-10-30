ntrains = []
with open('train_neg_0.85-855-151-1006.csv', 'r') as f:
    ntrains += f.readlines()
print(len(ntrains))
with open('train-nikko_neg_0.85-277-50-327.csv', 'r') as f:
    ntrains += f.readlines()   

print(len(ntrains))

with open('train_merge.csv', 'w') as f:
    f.write(''.join(ntrains))
    
    
ntests = []
with open('test_0.85-151.csv', 'r') as f:
    ntests += f.readlines()
print(len(ntests))
with open('test-nikko_0.85-50.csv', 'r') as f:
    ntests += f.readlines()   

print(len(ntests))

with open('test_merge.csv', 'w') as f:
    f.write(''.join(ntests))