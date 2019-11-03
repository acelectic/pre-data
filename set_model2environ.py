import os

os.environ['MODEL_RESNET50'] = 'evalresult/model-infer-merge-resnet50-ep20-loss-0.8572.h5'
print(os.environ['MODEL_RESNET50'])

os.environ['MODEL_RESNET101'] = 'evalresult/model-infer-merge-resnet101-ep20-loss-0.7293.h5'
print(os.environ['MODEL_RESNET101'])

os.environ['MODEL_cRESNET50'] = 'evalresult/model-infer-merge-resnet50-canchor-ep20-loss-0.1881.h5'
print(os.environ['MODEL_cRESNET50'])

os.environ['MODEL_cRESNET101'] = 'evalresult/model-infer-merge-resnet101-canchor-ep20-loss-0.3005.h5'
print(os.environ['MODEL_cRESNET101'])
