import albumentations.pytorch.transforms
import numpy                 as np

import torch
import cv2
import os
from collections import defaultdict
import random
import json

class MOTDetection(torch.utils.data.Dataset):
    def __init__(self, voc_root, annotation_filename, sample_transform=None):
        self.annotation_filename = annotation_filename
        self.sample_transform    = sample_transform
        self.root                = voc_root

        self.id_s = []
        self.im_dict = {}
        caption_set = set()
        with open(self.annotation_filename, 'r') as f:
            data = json.load(f)
        images = {img['id']: img for img in data['images']}
        annotations = data['annotations']
        for ann in annotations:
            captions = ann['captions']

            for caption in captions:
                caption_set.add(caption)

            image_id = ann['image_id']
            img_data = images[image_id]
            splits = img_data['file_name'].split('/')
            file_name = splits[0] + '/' + splits[2].split('_')[1] + '/' + splits[3].split('_')[0] + '/' + 'img1' + '/' + splits[3].split('_')[1]
            self.im_dict[image_id] = file_name

        caption_set.remove(None)
        caption_set = list(caption_set)

        self.caption_i2l = {i: caption_set[i] for i in range(len(caption_set))}
        self.caption_l2i = {value: key for key, value in self.caption_i2l.items()}

        self.im_labels_dict = defaultdict(list)
        for ann in data['annotations']:
            captions = []
            for caption in ann['captions']:
                if caption is not None:
                    captions.append(self.caption_l2i[caption])
            if len(captions) > 0:
                self.im_labels_dict[ann['image_id']].append((ann['bbox'], captions))
            #self.im_labels_dict[ann['image_id']].append((ann['bbox'], [*map(lambda x: self.caption_l2i[x] if x is not None else -1, ann['captions'])]))




        self.index_lst = []
        for i in range(len(self.im_labels_dict)):
            self.index_lst += [i] * len(self.im_labels_dict[i])

        self.label_lst = [0] * len(self.index_lst)
        counter = 0
        for i in range(1, len(self.index_lst)):
            if self.index_lst[i] != self.index_lst[i-1]:
                counter = 0
            else:
                counter += 1
            self.label_lst[i] = counter



    def __load_image(self, path):
        image = cv2.imread(path   , cv2.IMREAD_COLOR )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def __len__(self):
        return len(self.index_lst)

    def __getitem__(self, global_index):
        label_ind = self.label_lst[global_index]
        index = self.index_lst[global_index]

        while index not in self.im_dict:
            print(index)
            index += 1
            if index == len(self.index_lst):
                index = 0

        img_path = self.root + self.im_dict[index]
        ann_data = self.im_labels_dict[index]
        # label_set = []
        # for ann in ann_data:
        #     label_set += ann[1]
        label_set = set(ann_data[label_ind][-1])

        if -1 in label_set:
            label_set.remove(-1)

        label = random.sample(list(label_set), 1)[0]
        #label = list(label_set)[-1]
        #label = random.sample(list(label_set)[-2:], 1)[0]

        source_image = self.__load_image(img_path)
        image_width = source_image.shape[1]
        image_height = source_image.shape[0]

        source_box_s = []
        for ann in ann_data:
            if label in set(ann[1]):
                x_min, y_min, width, height = ann[0]
                x_max = x_min + width
                y_max = y_min + height
                normalized_bbox = [
                    x_min / image_width, y_min / image_height,
                    x_max / image_width, y_max / image_height
                ]
                normalized_bbox = np.clip(normalized_bbox, 0, 1)
                source_box_s.append(normalized_bbox)

        source_box_s = np.array(source_box_s)
        source_label_s = np.array([label] * source_box_s.shape[0])
        ready = False
        while not ready:
            try:
                result = self.sample_transform (image=source_image, bboxes=source_box_s, labels=source_label_s )
                ready = True
            except Exception as e:
                print('here2')
                global_index += 1
                label_ind = self.label_lst[global_index]
                index = self.index_lst[global_index]

                while index not in self.im_dict:
                    print(index)
                    index += 1
                    if index == len(self.index_lst):
                        index = 0

                img_path = self.root + self.im_dict[index]
                ann_data = self.im_labels_dict[index]
                # label_set = []
                # for ann in ann_data:
                #     label_set += ann[1]
                label_set = set(ann_data[label_ind][-1])

                if -1 in label_set:
                    label_set.remove(-1)
                label = random.sample(list(label_set), 1)[0]
                source_box_s = []
                source_image = self.__load_image(img_path)
                image_width = source_image.shape[1]
                image_height = source_image.shape[0]
                for ann in ann_data:
                    if label in set(ann[1]):
                        x_min, y_min, width, height = ann[0]
                        x_max = x_min + width
                        y_max = y_min + height
                        normalized_bbox = [
                            x_min / image_width, y_min / image_height,
                            x_max / image_width, y_max / image_height
                        ]
                        normalized_bbox = np.clip(normalized_bbox, 0, 1)
                        source_box_s.append(normalized_bbox)

                source_box_s = np.array(source_box_s)
                source_label_s = np.array([label] * source_box_s.shape[0])

        target_image   =          result['image' ]
        target_box_s   = np.array(result['bboxes'])
        target_label_s = np.array(result['labels'])


        #target_image   = torch.from_numpy( np.transpose( (source_image.astype(np.float32)/255), axes=(2,0,1) ))
        #target_box_s   = source_box_s
        #target_label_s = source_label_s

        return target_image.clone().detach(), torch.from_numpy(target_box_s), torch.from_numpy(target_label_s)


class MOTDetectionTest(torch.utils.data.Dataset):
    def __init__(self, voc_root, annotation_filename, sample_transform=None):
        self.annotation_filename = annotation_filename
        self.sample_transform    = sample_transform
        self.root                = voc_root

        self.id_s = []
        self.im_dict = {}
        caption_set = set()
        with open(self.annotation_filename, 'r') as f:
            data = json.load(f)
        images = {img['id']: img for img in data['images']}
        annotations = data['annotations']
        for ann in annotations:
            captions = ann['captions']

            for caption in captions:
                caption_set.add(caption)

            image_id = ann['image_id']
            img_data = images[image_id]
            splits = img_data['file_name'].split('/')
            file_name = splits[0] + '/' + splits[2].split('_')[1] + '/' + splits[3].split('_')[0] + '/' + 'img1' + '/' + splits[3].split('_')[1]
            self.im_dict[image_id] = file_name

        caption_set.remove(None)
        caption_set = list(caption_set)

        self.caption_i2l = {i: caption_set[i] for i in range(len(caption_set))}
        self.caption_l2i = {value: key for key, value in self.caption_i2l.items()}

        self.im_labels_dict = defaultdict(list)
        for ann in data['annotations']:
            captions = []
            for caption in ann['captions']:
                if caption is not None:
                    captions.append(self.caption_l2i[caption])
            if len(captions) > 0:
                self.im_labels_dict[ann['image_id']].append((ann['bbox'], captions))
            #self.im_labels_dict[ann['image_id']].append((ann['bbox'], [*map(lambda x: self.caption_l2i[x] if x is not None else -1, ann['captions'])]))




        self.index_lst = []
        for i in range(len(self.im_labels_dict)):
            self.index_lst += [i] * len(self.im_labels_dict[i])

        self.label_lst = [0] * len(self.index_lst)
        counter = 0
        for i in range(1, len(self.index_lst)):
            if self.index_lst[i] != self.index_lst[i-1]:
                counter = 0
            else:
                counter += 1
            self.label_lst[i] = counter



    def __load_image(self, path):
        image = cv2.imread(path   , cv2.IMREAD_COLOR )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def __len__(self):
        return 1000

    def __getitem__(self, global_index):
        label_ind = self.label_lst[global_index]
        index = self.index_lst[global_index]

        while index not in self.im_dict:
            print(index)
            index += 1
            if index == len(self.index_lst):
                index = 0

        img_path = self.root + self.im_dict[index]
        ann_data = self.im_labels_dict[index]
        # label_set = []
        # for ann in ann_data:
        #     label_set += ann[1]
        label_set = set(ann_data[label_ind][-1])

        if -1 in label_set:
            label_set.remove(-1)

        label = random.sample(list(label_set), 1)[0]
        #label = list(label_set)[-1]
        #label = random.sample(list(label_set)[-2:], 1)[0]

        source_image = self.__load_image(img_path)
        image_width = source_image.shape[1]
        image_height = source_image.shape[0]

        source_box_s = []
        for ann in ann_data:
            if label in set(ann[1]):
                x_min, y_min, width, height = ann[0]
                x_max = x_min + width
                y_max = y_min + height
                normalized_bbox = [
                    x_min / image_width, y_min / image_height,
                    x_max / image_width, y_max / image_height
                ]
                normalized_bbox = np.clip(normalized_bbox, 0, 1)
                source_box_s.append(normalized_bbox)

        source_box_s = np.array(source_box_s)
        source_label_s = np.array([label] * source_box_s.shape[0])
        ready = False
        while not ready:
            try:
                result = self.sample_transform (image=source_image, bboxes=source_box_s, labels=source_label_s )
                ready = True
            except Exception as e:
                print('here2')
                global_index += 1
                label_ind = self.label_lst[global_index]
                index = self.index_lst[global_index]

                while index not in self.im_dict:
                    print(index)
                    index += 1
                    if index == len(self.index_lst):
                        index = 0

                img_path = self.root + self.im_dict[index]
                ann_data = self.im_labels_dict[index]
                # label_set = []
                # for ann in ann_data:
                #     label_set += ann[1]
                label_set = set(ann_data[label_ind][-1])

                if -1 in label_set:
                    label_set.remove(-1)
                label = random.sample(list(label_set), 1)[0]
                source_box_s = []
                source_image = self.__load_image(img_path)
                image_width = source_image.shape[1]
                image_height = source_image.shape[0]
                for ann in ann_data:
                    if label in set(ann[1]):
                        x_min, y_min, width, height = ann[0]
                        x_max = x_min + width
                        y_max = y_min + height
                        normalized_bbox = [
                            x_min / image_width, y_min / image_height,
                            x_max / image_width, y_max / image_height
                        ]
                        normalized_bbox = np.clip(normalized_bbox, 0, 1)
                        source_box_s.append(normalized_bbox)

                source_box_s = np.array(source_box_s)
                source_label_s = np.array([label] * source_box_s.shape[0])

        target_image   =          result['image' ]
        target_box_s   = np.array(result['bboxes'])
        target_label_s = np.array(result['labels'])


        #target_image   = torch.from_numpy( np.transpose( (source_image.astype(np.float32)/255), axes=(2,0,1) ))
        #target_box_s   = source_box_s
        #target_label_s = source_label_s

        return target_image.clone().detach(), torch.from_numpy(target_box_s), torch.from_numpy(target_label_s)


class OmniDetection(torch.utils.data.Dataset):
    def __init__(self, voc_root, annotation_filename, sample_transform=None):
        self.annotation_filename = annotation_filename
        self.sample_transform    = sample_transform
        self.root                = voc_root

        with open(annotation_filename, 'r') as f:
            data = json.load(f)
        self.im_dict = {img['id']: img['file_name'] for img in data['images']}

        self.caption_set = set()

        for desc in data['descriptions']:
            self.caption_set.add(desc['text'])

        self.caption_set = sorted(list(self.caption_set))

        self.caption_i2l = {i: self.caption_set[i] for i in range(len(self.caption_set))}
        self.caption_l2i = {value: key for key, value in self.caption_i2l.items()}

        self.desc_dct = {}
        for el in data['descriptions']:
            self.desc_dct[el['id']] = el['text']

        self.im_labels_dict = defaultdict(list)
        for ann in data['annotations']:
            self.im_labels_dict[ann['image_id']].append((ann['bbox'], [self.caption_l2i[self.desc_dct[ann['description_ids'][-1]]]])) #TODO: multiple descriptions

        self.index_lst = []
        # for i in range(len(self.im_labels_dict)):
        #     self.index_lst += [i] * len(self.im_labels_dict[i])

        for i in list(self.im_labels_dict.keys()):
            self.index_lst += [i] * len(self.im_labels_dict[i])

        self.label_lst = [0] * len(self.index_lst)
        counter = 0
        for i in range(1, len(self.index_lst)):
            if self.index_lst[i] != self.index_lst[i-1]:
                counter = 0
            else:
                counter += 1
            self.label_lst[i] = counter


    def __load_image(self, path):
        image = cv2.imread(path   , cv2.IMREAD_COLOR )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def __len__(self):
        return len(self.index_lst)

    def __getitem__(self, global_index):
        label_ind = self.label_lst[global_index]
        index = self.index_lst[global_index]

        while index not in self.im_dict:
            print(index)
            index += 1
            if index == len(self.index_lst):
                index = 0

        if 'object365' in self.im_dict[index]:
            img_path = self.root + 'object365/' + self.im_dict[index].split('/')[-1]

        else:
            img_path = self.root + self.im_dict[index]
        #print(img_path, index)
        #print()

        ann_data = self.im_labels_dict[index]
        # label_set = []
        # for ann in ann_data:
        #     label_set += ann[1]
        label_set = set(ann_data[label_ind][-1])

        if -1 in label_set:
            label_set.remove(-1)

        label = random.sample(list(label_set), 1)[0]
        #label = list(label_set)[-1]
        #label = random.sample(list(label_set)[-2:], 1)[0]

        source_image = self.__load_image(img_path)
        image_width = source_image.shape[1]
        image_height = source_image.shape[0]

        source_box_s = []
        for ann in ann_data:
            if label in set(ann[1]):
                x_min, y_min, width, height = ann[0]
                x_max = x_min + width
                y_max = y_min + height
                normalized_bbox = [
                    x_min / image_width, y_min / image_height,
                    x_max / image_width, y_max / image_height
                ]
                normalized_bbox = np.clip(normalized_bbox, 0, 1)
                source_box_s.append(normalized_bbox)

        source_box_s = np.array(source_box_s)
        source_label_s = np.array([label] * source_box_s.shape[0])
        ready = False
        while not ready:
            try:
                result = self.sample_transform (image=source_image, bboxes=source_box_s, labels=source_label_s )
                ready = True
            except Exception as e:
                print('here2')
                global_index += 1
                label_ind = self.label_lst[global_index]
                index = self.index_lst[global_index]

                while index not in self.im_dict:
                    print(index)
                    index += 1
                    if index == len(self.index_lst):
                        index = 0

                img_path = self.root + self.im_dict[index]
                ann_data = self.im_labels_dict[index]
                # label_set = []
                # for ann in ann_data:
                #     label_set += ann[1]
                label_set = set(ann_data[label_ind][-1])

                if -1 in label_set:
                    label_set.remove(-1)
                label = random.sample(list(label_set), 1)[0]
                source_box_s = []
                source_image = self.__load_image(img_path)
                image_width = source_image.shape[1]
                image_height = source_image.shape[0]
                for ann in ann_data:
                    if label in set(ann[1]):
                        x_min, y_min, width, height = ann[0]
                        x_max = x_min + width
                        y_max = y_min + height
                        normalized_bbox = [
                            x_min / image_width, y_min / image_height,
                            x_max / image_width, y_max / image_height
                        ]
                        normalized_bbox = np.clip(normalized_bbox, 0, 1)
                        source_box_s.append(normalized_bbox)

                source_box_s = np.array(source_box_s)
                source_label_s = np.array([label] * source_box_s.shape[0])

        target_image   =          result['image' ]
        target_box_s   = np.array(result['bboxes'])
        target_label_s = np.array(result['labels'])


        #target_image   = torch.from_numpy( np.transpose( (source_image.astype(np.float32)/255), axes=(2,0,1) ))
        #target_box_s   = source_box_s
        #target_label_s = source_label_s

        return target_image.clone().detach(), torch.from_numpy(target_box_s), torch.from_numpy(target_label_s)

