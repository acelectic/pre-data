import os

train = 'train_0.85-1885.csv'
train_neg = 'train_neg_0.85-1885-332-2217.csv'

train_file = train
# train_file = train_neg

with open(train_file) as f:
  print(len(f.readlines()))
  train_size = len(f.readlines())
  f.close


test_file = 'test_0.85-333.csv'
class_file = 'classes'

weights = "resnet50_coco_best_v2.1.0.h5"

max_side=700
min_side=700

epochs= 10
#epochs= 1

batch_size=1

steps=int(train_size//1)
steps= 2000
print(steps)

cmds = {}
base_path = os.getcwd() +'/store_models'

def create_dir(path):
  snapshot_path = base_path +'/'+ path + '/snapshots'
  tensorboard_dir = base_path +'/'+ path + '/logs'

  # Create target directory & all intermediate directories if don't exists
  try:
      os.makedirs(snapshot_path)    
      print("Directory " , snapshot_path ,  " Created ")
  except FileExistsError:
      print("Directory " , snapshot_path ,  " already exists")  

  try:
      os.makedirs(tensorboard_dir)    
      print("Directory " , tensorboard_dir ,  " Created ")
  except FileExistsError:
      print("Directory " , tensorboard_dir ,  " already exists") 

  return snapshot_path, tensorboard_dir
      

def gen_command(train_file=None, test_file=None, class_file=None,  weights= None,
    backbone = backbone, max_side= 700, min_side = 700, batch_size= 1, epochs=10, steps= 1000,
    anchor_config = None, snapshot_path=None, tensorboard_dir=None, store_path=None):
  snapshot_path, tensorboard_dir = create_dir(store_path)

  cmd = "retinanet-train --backbone {backbone}\
    --weights {weights} --image-max-side {max} --image-min-side {min}\
    --batch-size {batch_size} --epochs {epochs} --steps {steps} --weighted-average --compute-val-loss\
      {snapshot_path} {tensorboard_dir} {anchor_config}\
      csv {train_file} {class_file} --val-annotations {test_file} "\
    .format(backbone='"'+backbone+'"', max = max_side, min=min_side,
        batch_size=batch_size, epochs=epochs, steps=steps,
        weights=weights or "resnet50_coco_best_v2.1.0.h5",
        train_file=train_file or 'train_neg_0.85-1885-332-2217.csv',
        class_file= class_file or 'classes',
        test_file=test_file or 'test_0.85-333.csv',
        anchor_config = '--config ' + anchor_config if anchor_config else '',
        snapshot_path = '--snapshot-path ' + snapshot_path if snapshot_path else '',
        tensorboard_dir = '--tensorboard-dir ' + tensorboard_dir if tensorboard_dir else '')
  return cmd

def test_cmd():
  return gen_command(epochs=1, steps=10)


### resnet50 700 700
backbone = 'resnet50'
#	default anchor
cmd = gen_command(train_file= train_file,
                  test_file= test_file,
                  class_file= class_file,
                  weights=weights,
                  store_path='default',
                  backbone = backbone, max_side= max_side, min_side = min_side, batch_size= batch_size, epochs=epochs, steps= steps)

cmds['0'] = {'name':'default', 'cmd':cmd}

#	anchor	3r 
anchor_3r = 'retinanet-3.ini'
cmd = gen_command(train_file= train_file,
                  test_file= test_file,
                  class_file= class_file,
                  weights=weights,
                  anchor_config=anchor_3r,
                  store_path='anchor_3r',
                  backbone = backbone, max_side= max_side, min_side = min_side, batch_size= batch_size, epochs=epochs, steps= steps)

cmds['1'] = {'name':'anchor_3r', 'cmd':cmd}


#	anchor	5r 3s 
anchor_5r_3s = 'retina-5r-3s.ini'
cmd = gen_command(train_file= train_file,
                  test_file= test_file,
                  class_file= class_file,
                  weights=weights,
                  anchor_config=anchor_5r_3s,
                  store_path='anchor_5r_3s',
                  backbone = backbone, max_side= max_side, min_side = min_side, batch_size= batch_size, epochs=epochs, steps= steps)

cmds['2'] = {'name':'anchor_5r_3s', 'cmd':cmd}


### resnet101 700 700
backbone = 'resnet101'
#	default anchor
cmd = gen_command(train_file= train_file,
                  test_file= test_file,
                  class_file= class_file,
                  weights=weights,
                  store_path='default',
                  backbone = backbone, max_side= max_side, min_side = min_side, batch_size= batch_size, epochs=epochs, steps= steps)

cmds['0'] = {'name':'default', 'cmd':cmd}

#	anchor	3r 
anchor_3r = 'retinanet-3.ini'
cmd = gen_command(train_file= train_file,
                  test_file= test_file,
                  class_file= class_file,
                  weights=weights,
                  anchor_config=anchor_3r,
                  store_path='anchor_3r',
                  backbone = backbone, max_side= max_side, min_side = min_side, batch_size= batch_size, epochs=epochs, steps= steps)

cmds['1'] = {'name':'anchor_3r', 'cmd':cmd}


#	anchor	5r 3s 
anchor_5r_3s = 'retina-5r-3s.ini'
cmd = gen_command(train_file= train_file,
                  test_file= test_file,
                  class_file= class_file,
                  weights=weights,
                  anchor_config=anchor_5r_3s,
                  store_path='anchor_5r_3s',
                  backbone = backbone, max_side= max_side, min_side = min_side, batch_size= batch_size, epochs=epochs, steps= steps)

cmds['2'] = {'name':'anchor_5r_3s', 'cmd':cmd}

if __name__ == '__main__':
  run = True
  #run = False
  # for i in cmds:
  #   print("{i}\n{cmd}".format(i=i, cmd=cmds[i]))
    
  if run:
      tmp_cmd = cmds['0']
      print('{}\n\n{}\n\n{}'.format('#'*30, tmp_cmd['name'], '#'*30))
      os.system(tmp_cmd['cmd']) 