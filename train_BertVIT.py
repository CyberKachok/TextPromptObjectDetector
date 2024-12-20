from lib.models.BertVIT.train_utils import *
from lib.models.BertVIT.BVDetector import BVPromptDetector
from lib.models.BertVIT.train_utils import train_process
from lib.utils.MOTdataset import OmniDetection
from torch.utils.data import DataLoader
from argparse import Namespace
import albumentations as A

root_dir = '/media/ilya/FastDisk/Datasets/Omnilabels/data/'
ann_path = '/media/ilya/FastDisk/Datasets/Omnilabels/dataset_all_val_v0.1.4.json'

custom_config = {
    'num_classes': 2,
    'feature_maps': [(31, 31), (16, 16), (8, 8), (5, 5), (2, 2)],

    'min_sizes': [0.10, 0.20, 0.37, 0.62, 0.84],
    'max_sizes': [0.20, 0.37, 0.62, 0.84, 1.05],
    'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2]],
    'num_priors': [6, 6, 6, 6, 4],
    'variance': [0.1, 0.2],
    'clip': True,

    'overlap_threshold': 0.25,
    'neg_pos_ratio': 3,

    'model_name': 'test'
}

param_s = Namespace(
    epochs=400, batch_size=24,
    checkpoint=None, output='output',
    learning_rate=5e-3, momentum=0.9,
    weight_decay=0.00005,
    num_workers=30,
    seed=0
)

model = BVPromptDetector()
model = model.cuda().train()

SIZE = 320
bbox_params = A.BboxParams(format='albumentations', min_area=0, min_visibility=0.0, label_fields=['labels'])

light_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    #A.RandomResizedCrop(height=257, width=257, p=0.5, scale=(0.7, 1.0)),
    A.Resize(height=SIZE, width=SIZE),
    A.GaussNoise(var_limit=(100, 150), p=0.5),
    A.RGBShift(p=0.5),
    A.Blur(blur_limit=11, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.CLAHE(p=0.5),
    A.augmentations.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), always_apply=True),
    A.pytorch.transforms.ToTensorV2()
], bbox_params=bbox_params, p=1.0)

test_transform = A.Compose([
    #A.HorizontalFlip(p=0.5),
    #A.RandomResizedCrop(height=257, width=257, p=0.5, scale=(0.7, 1.0)),
    A.Resize(height=SIZE, width=SIZE),
    A.augmentations.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), always_apply=True),
    A.pytorch.transforms.ToTensorV2()
], bbox_params=bbox_params, p=1.0)


def default_collate(batch):
    label_ss, box_ss, image_s = [], [], []

    for sample in batch:
        image, box_s, label_s = sample

        if len(box_s) > 0 and len(label_s) > 0:
            image_s.append(image)
            box_ss.append(box_s)
            label_ss.append(label_s)

    return torch.stack(image_s) if len(image_s) > 0 else torch.Tensor(), box_ss, label_ss


data = OmniDetection(root_dir, ann_path, light_transform)
n_tr = int(len(data) * 0.8)

train_set, val_set = torch.utils.data.random_split(data, [n_tr, len(data) - n_tr],
                                                   generator=torch.Generator().manual_seed(param_s.seed))

val_set.dataset.sample_transform = test_transform

dataloaders = {'train': DataLoader(train_set, num_workers=param_s.num_workers, batch_size=param_s.batch_size, shuffle=True, collate_fn=default_collate,
                                   drop_last=True),
               'test': DataLoader(val_set, num_workers=param_s.num_workers, batch_size=param_s.batch_size, shuffle=False, collate_fn=default_collate), }

model, prior_box_s, train_loss_s, eval_loss_s = train_process(model, dataloaders, param_s, custom_config)
