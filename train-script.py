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

max_side = 700
min_side = 700

epochs = 10
#epochs= 1

batch_size = 1

steps = int(train_size//1)
steps = 10
print(steps)

cmds = {}
base_path = os.getcwd() + '/store_models/main'
base_path = os.getcwd() + '/store_models/test'

default_args = {
    'train_file': train_file,
    'test_file': test_file,
    'class_file': class_file,
    'weights': "resnet50_coco_best_v2.1.0.h5",
    'epochs': epochs,
    'batch_size': batch_size,
    'steps': steps,
}

resnet50_params = {
    'backbone': "resnet50",
    'max_side': 700,
    'min_side': 700,
}

resnet101_params = {
    'backbone': "resnet101",
    'max_side': 400,
    'min_side': 400,
}

def export_inference_model():
    retinanet-convert-model training-model.h5 inference-model.h5

def create_dir(backbone, path):
    snapshot_path = base_path + '/'+backbone+'/' + path + '/snapshots'
    tensorboard_dir = base_path + '/'+backbone+'/' + path + '/logs'

    # Create target directory & all intermediate directories if don't exists
    try:
        os.makedirs(snapshot_path)
        print("Directory ", snapshot_path,  " Created ")
    except FileExistsError:
        print("Directory ", snapshot_path,  " already exists")

    try:
        os.makedirs(tensorboard_dir)
        print("Directory ", tensorboard_dir,  " Created ")
    except FileExistsError:
        print("Directory ", tensorboard_dir,  " already exists")

    return snapshot_path, tensorboard_dir


def optimize_anchor():
    cmd = "python3 optimize_anchors.py '/home/minibear/Desktop/pre-data/pigeon.csv' --no-resize --ratios=5 --image-max-side 700 --image-min-side 700"


def debug():
    cmd = "retinanet-debug --anchors --display-name --annotations --config retina-5r-3s.ini csv train_neg_0.85-1885-332-2217.csv classes"


def gen_command(train_file=None, test_file=None, class_file=None,  weights=None,
                backbone="resnet50", max_side=700, min_side=700, batch_size=1, epochs=10, steps=1000,
                anchor_config=None, snapshot_path=None, tensorboard_dir=None, store_path=None, snapshot=None):

    snapshot_path, tensorboard_dir = create_dir(backbone, store_path)

    cmd = "retinanet-train {snapshot} --backbone {backbone}\
     {weights} --image-max-side {max} --image-min-side {min}\
    --batch-size {batch_size} --epochs {epochs} --steps {steps} --weighted-average --compute-val-loss\
      {snapshot_path} {tensorboard_dir} {anchor_config}\
      csv {train_file} {class_file} --val-annotations {test_file} "\
      .format(backbone='"'+backbone+'"', max=max_side, min=min_side,
              batch_size=batch_size, epochs=epochs, steps=steps,
              weights='--weights ' +
              weights or "--weights resnet50_coco_best_v2.1.0.h5" if not snapshot else '',
              train_file=train_file or 'train_neg_0.85-1885-332-2217.csv',
              class_file=class_file or 'classes',
              test_file=test_file or 'test_0.85-333.csv',
              anchor_config='--config ' + anchor_config if anchor_config else '',
              snapshot_path='--snapshot-path ' + snapshot_path if snapshot_path else '',
              tensorboard_dir='--tensorboard-dir ' +
              tensorboard_dir if tensorboard_dir else '',
              snapshot='--snapshot '+snapshot if snapshot else '')

    args = {
        'train_file': train_file,
        'test_file': test_file,
        'class_file': class_file,
        'weights': weights,
        'backbone': backbone,
        'max_side': max_side,
        'min_side': min_side,
        'batch_size': batch_size,
        'epochs': epochs,
        'steps': steps,
        'anchor_config': anchor_config,
        'snapshot_path': snapshot_path,
        'tensorboard_dir': tensorboard_dir,
        'store_path': store_path,
        'snapshot': snapshot
    }
    return cmd, args


def test_cmd():
    return gen_command(epochs=1, steps=10)


# resnet50 700 700
backbone = 'resnet50'
#	default anchor
cmd, args = gen_command(store_path='default',
                        backbone=backbone, **default_args)

cmds['0'] = {'name': 'default', 'cmd': cmd, 'backbone': backbone, 'args': args}

#	anchor	3r
anchor_3r = 'retinanet-3.ini'
cmd, args = gen_command(anchor_config=anchor_3r,
                        store_path='anchor_3r',
                        backbone=backbone, **default_args)

cmds['1'] = {'name': 'anchor_3r', 'cmd': cmd,
             'backbone': backbone, 'args': args}


#	anchor	5r 3s
anchor_5r_3s = 'retina-5r-3s.ini'
cmd, args = gen_command(anchor_config=anchor_5r_3s,
                        store_path='anchor_5r_3s',
                        backbone=backbone, **default_args)

cmds['2'] = {'name': 'anchor_5r_3s', 'cmd': cmd,
             'backbone': backbone, 'args': args}

default_args = {
    'train_file': train_file,
    'test_file': test_file,
    'class_file': class_file,
    'weights': "resnet50_coco_best_v2.1.0.h5",
    'max_side': 400,
    'min_side': 400,
    'epochs': epochs,
    'batch_size': batch_size,
    'steps': steps,
}

# resnet101 300 300
backbone = 'resnet101'
#	default anchor
cmd, args = gen_command(store_path='default',
                        snapshot='store_models/resnet101/default/snapshots/resnet101_01_loss-26.9501_val-loss-2.5892_mAP-0.3225.h5',
                        backbone=backbone, **default_args)

cmds['3'] = {'name': 'default', 'cmd': cmd, 'backbone': backbone, 'args': args}

#	anchor	3r
anchor_3r = 'retinanet-3.ini'
cmd, args = gen_command(anchor_config=anchor_3r,
                        store_path='anchor_3r',
                        backbone=backbone, **default_args)

cmds['4'] = {'name': 'anchor_3r', 'cmd': cmd,
             'backbone': backbone, 'args': args}


#	anchor	5r 3s snapshot
anchor_5r_3s = 'retina-5r-3s.ini'
cmd, args = gen_command(anchor_config=anchor_5r_3s,
                        store_path='anchor_5r_3s',
                        backbone=backbone, **default_args)
cmds['5'] = {'name': 'anchor_5r_3s', 'cmd': cmd,
             'backbone': backbone, 'args': args}

if __name__ == '__main__':
    run = True
    #run = False
    # for i in cmds:
    #   print("{i}\n{cmd}".format(i=i, cmd=cmds[i]))

    if run:
        tmp_cmd = cmds['1']
        print('{s}\n\nname: {name}\nbackbone: {backbone}\n\n{args}\n\n{n}'.format(s='#'*30, n='#'*30,
                                                                                name=tmp_cmd['name'],
                                                                                backbone=tmp_cmd['backbone'],
                                                                                args='\n'.join(sorted(['{: <20}{}'.format(k, str(v)) for k, v in tmp_cmd['args'].items()]))))
        os.system(tmp_cmd['cmd'])
