
images = open('pigeon-merge-nikko.csv', 'r').readlines()

image_dicts = {}
bbox_total = len(images)

for i in images:
    name_img = i.split(',')[0]
    try:
        image_dicts[name_img] += 1
    except:
        image_dicts[name_img] = 1


print('images: {}\nbboxs: {}'.format(len(image_dicts) ,bbox_total))

images = open('train_merge.csv', 'r').readlines()

image_dicts = {}
bbox_total = len(images)

for i in images:
    name_img = i.split(',')[0]
    try:
        image_dicts[name_img] += 1
    except:
        image_dicts[name_img] = 1


print('images: {}\nbboxs: {}'.format(len(image_dicts) ,bbox_total))

images = open('test_merge.csv', 'r').readlines()

image_dicts = {}
bbox_total = len(images)

for i in images:
    name_img = i.split(',')[0]
    try:
        image_dicts[name_img] += 1
    except:
        image_dicts[name_img] = 1


print('images: {}\nbboxs: {}'.format(len(image_dicts) ,bbox_total))