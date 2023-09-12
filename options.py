from easydict import EasyDict

img_h = 320
img_w = 640

OPTION = EasyDict()

# ------------------------------------------ data configuration ---------------------------------------------
OPTION.trainset = ['VDS']
OPTION.valset = 'VDS'
OPTION.setting = '60_lr0.001deay1e-6_sgd'
OPTION.root = './dataset'  # dataset root path
OPTION.datafreq = 4 #
OPTION.max_object = 8 # max number of instances
OPTION.input_size = (img_h, img_w)   # input image size
OPTION.sampled_frames = 8 #9       # min sampled time length while trianing
OPTION.max_skip = [5]         # max skip time length while trianing
OPTION.samples_per_video = 2  # sample numbers per video

# ----------------------------------------- model configuration ---------------------------------------------
OPTION.keydim = 128
OPTION.valdim = 512
OPTION.save_freq = 1
OPTION.save_freq_max = 100
OPTION.epochs_per_increment = 2

# ---------------------------------------- training configuration -------------------------------------------
OPTION.epochs = 60 #60
OPTION.train_batch = 8#4
OPTION.learning_rate = 2e-4 #1e-3 记得改！
OPTION.gamma = 0.1
OPTION.momentum = (0.9, 0.999)
OPTION.solver = 'adamW'             # 'sgd' or 'adam' 'adamW'
OPTION.weight_decay = 5e-4  #5e-4
OPTION.iter_size = 1
OPTION.milestone = []              # epochs to degrades the learning rate
OPTION.loss = 'LIoU'               # 'ce' or 'iou' or 'both' 'dice'
OPTION.mode = 'few_shot'           #
OPTION.iou_threshold = 0.65        # used only for 'threshold' training

# ---------------------------------------- testing configuration --------------------------------------------
OPTION.epoch_per_test = 5

# ------------------------------------------- other configuration -------------------------------------------
OPTION.checkpoint = 'models'
OPTION.initial = ''      # path to initialize the backbone
OPTION.initial_featNet = './models/featModel/1009_warp_60.pth.tar'
OPTION.initial_taskNet = './models/taskModel/withoutTR.pth.tar'
OPTION.initial_model = ''
# path to restart from the checkpoint
OPTION.resume_featNet = ''#'./models/featModel/1009_warp_60.pth.tar'
OPTION.resume_taskNet = ''#'./models/taskModel/few_shot_model_best.pth.tar'
OPTION.resume_model = ''
OPTION.gpu_id = '0'      # defualt gpu-id (if not specified in cmd)
OPTION.workers = 0       #1
OPTION.save_indexed_format = True # set True to save indexed format png file, otherwise segmentation with original image
OPTION.output_dir = './output'

#-----------------------------
OPTION.transforms = [
            dict(name='Resize',
                 parameters=dict(size=dict(height=img_h, width=img_w)),
                 p=1.0),
            dict(name='HorizontalFlip', parameters=dict(p=1.0), p=0.1),
            dict(name='ChannelShuffle', parameters=dict(p=1.0), p=0.1),
            dict(name='MultiplyAndAddToBrightness',
                 parameters=dict(mul=(0.85, 1.15), add=(-10, 10)),
                 p=0.6),
            dict(name='AddToHueAndSaturation',
                 parameters=dict(value=(-10, 10)),
                 p=0.7),
            dict(name='OneOf',
                 transforms=[
                     dict(name='MotionBlur', parameters=dict(k=(3, 5))),
                     dict(name='MedianBlur', parameters=dict(k=(3, 5)))
                 ],
                 p=0.2),
            dict(name='Affine',
                 parameters=dict(translate_percent=dict(x=(-0.1, 0.1),
                                                        y=(-0.1, 0.1)),
                                 rotate=(-3, 3),
                                 scale=(0.95, 1.05)),
                 p=0.5),
            dict(name='Resize',
                 parameters=dict(size=dict(height=img_h, width=img_w)),
                 p=1.0),
        ]

OPTION.test_transforms = [
             dict(name='Resize',
                  parameters=dict(size=dict(height=img_h, width=img_w)),
                  p=1.0),
         ]

OPTION.img_h = img_h
OPTION.img_w = img_w
OPTION.num_points = 72
OPTION.n_offsets = 72
OPTION.max_lanes = 8
OPTION.cut_scale = 0.35