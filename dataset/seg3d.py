import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import SimpleITK as sitk
import random

class SentenceBuilder:
    def age_description(self, age, digit=False):
        if digit:
            return str(age)
        if age < 25:
            return 'a young'
        elif 25 <= age < 35:
            return 'a young-adult'
        elif 35 <= age < 45:
            return 'an adult'
        elif 45 <= age < 55:
            return 'a middle-aged'
        elif 55 <= age < 65:
            return 'an elderly'
        elif 65 <= age < 75:
            return 'a senior'
        elif 75 <= age < 85:
            return 'an advanced-senior'
        elif age >= 85:
            return 'a long-lived-senior'
        else:
            return 'an unknown-age'

    def gender_description(self, gender):
        male_descriptions = ['male', 'man', 'gentleman']
        female_descriptions = ['female', 'woman', 'lady']

        if gender == 'm':
            # 随机选择一个男性描述词
            return random.choice(male_descriptions)
        elif gender == 'f':
            # 随机选择一个女性描述词
            return random.choice(female_descriptions)
        else:
            return 'unspecified-gender'

    def plane_full_name(self, plane, short=False):
        if short:
            return plane
        mapping = {'sag': 'sagittal', 'hor': 'horizontal', 'cor': 'coronal'}
        
        return mapping.get(plane, 'unknown')

    def spine_full_name(self, spine, short=False):
        if short:
            return spine
        spine_mapping = {
            'C': 'Cervical',
            'T': 'Thoracic',
            'L': 'Lumbar',
            'S': 'Sacrum',
        }
        try:
            tag = spine[0]
            name = spine_mapping[tag]
            if spine[1] == '0':
                return name
            else:
                return name + '-' + spine[1:]
        except:
            return spine

    def list_to_natural_language(self, lst):
        if not lst:
            return 'none'
        elif len(lst) == 1:
            return lst[0]
        else:
            return ', '.join(lst[:-1]) + ' and ' + lst[-1]
        
    def order_description(self, positive_order):
        if positive_order:
            return 'in a forward sequence'
        else: 
            return 'in a reverse sequence'

    def dict_to_sentence(self, d, randomize=True):
        if randomize:
            age_digit = random.choice([True, False])
            short_spine = random.choice([True, False])
            short_plane = random.choice([True, False])
        else:
            age_digit = False
            short_spine = False
            short_plane = False
        
        parts = {
            'age': self.age_description(d['age'],digit=age_digit) if 'age' in d else "",
            'gender': self.gender_description(d['gender']) if 'gender' in d else "",
            'plane': self.plane_full_name(d['plane'], short=short_plane) if 'plane' in d else "",
            'spine_list': self.list_to_natural_language([self.spine_full_name(spine, short=short_spine) for spine in d['spines'].split('|')]) if 'spines' in d else "",
            'order': self.order_description(d['positive_order']),
        }
        
        if not randomize:
            output = ''
            for k, v in parts.items():
                if v != '':
                    output += f"{k}:{v}, "
            return output[:-2]

        templates = [
            f"{parts['age']} {parts['gender']} patient has observations {parts['order']} in the {parts['plane']} plane, including {parts['spine_list']}.",
            f"In the {parts['plane']} plane {parts['order']}, the {parts['gender']} patient of {d.get('age', 'unknown-age')}-years shows the following spines: {parts['spine_list']}.",
            f"For {parts['age']} {parts['gender']}, the CT scan in the {parts['plane']} plane {parts['order']} reveals {parts['spine_list']}.",
            f"Observations for {parts['age']} {parts['gender']} {parts['order']}: {parts['spine_list']} in the {parts['plane']} plane.",
            f"{parts['age']} {parts['gender']} shows {parts['spine_list']} on {parts['plane']} plane imaging {parts['order']}.",
            f"{parts['plane'].capitalize()} plane analysis {parts['order']} reveals {parts['spine_list']} for this {parts['age']} {parts['gender']}.",
            f"CT findings: {parts['spine_list']} in the {parts['plane']} plane {parts['order']} for {parts['age']} {parts['gender']}.",
            f"Diagnosis for {parts['age']} {parts['gender']}: {parts['spine_list']}, as seen in the {parts['plane']} plane.",
            f"{parts['plane'].capitalize()} plane imaging {parts['order']} {('reveals ' + parts['spine_list']) if parts['spine_list'] else 'analysis'} for {(parts['age'] + ' ') if parts['age'] else ''}{parts['gender']} patient.",
            f"Patient details: {(', '.join(filter(None, [parts['age'], parts['gender'], parts['spine_list']])) + ',') if any([parts['age'], parts['gender'], parts['spine_list']]) else 'Not fully specified'}, observed in {parts['plane']} plane {parts['order']}.",
            f"{(parts['age'] + ' ') if parts['age'] else ''}{parts['gender']} with findings in the {parts['plane']} plane {parts['order']}: {parts['spine_list']}.",
            f"{('Findings for ' + parts['age'] + ' ') if parts['age'] else ''}{parts['gender']}: {parts['spine_list']} in the {parts['plane']} plane {parts['order']}.",
            f"CT scan {('in ' + parts['plane'] + ' plane ') if parts['plane'] else ''}{parts['order']} shows {parts['spine_list']} for {(parts['age'] + ' ') if parts['age'] else ''}{parts['gender']}.",
            f"Analysis{(': ' + parts['plane'] + ' plane, ') if parts['plane'] else ': '} {parts['order']} {parts['spine_list']} {'for ' + parts['age'] + ' ' + parts['gender'] if parts['age'] and parts['gender'] else ('for ' + (parts['age'] or parts['gender']))}.",
            f"{parts['age']} {parts['gender']}, {parts['plane']}: {parts['spine_list']}.",
            f"{parts['gender']} {parts['age']}, {parts['spine_list']} in {parts['plane']} {parts['order']}.",
            f"{parts['plane']} {parts['order']} - {parts['spine_list']}, {parts['age']} {parts['gender']}.",
            f"{parts['age']} {parts['gender']} {parts['plane']}: {parts['spine_list']}.",
            f"{parts['spine_list']} ({parts['plane']}, {parts['age']} {parts['gender']}) {parts['order']}.",
            f"{parts['age']} {parts['gender']}, {parts['plane']} plane {parts['order']}.",
            f"{parts['gender']} patient, {parts['spine_list']} {parts['order']}.",
            f"{parts['plane']} plane: {parts['spine_list']}.",
            f"{parts['age']} {parts['gender']}: {parts['spine_list']} {parts['order']}.",
            f"{parts['spine_list']}, {parts['plane']} plane {parts['order']}.",
            ""
        ]

        # 随机选择一个模板并返回
        return random.choice(templates).replace("  ", " ").replace(",,", ",").strip()
    
class Sag3DDataset(SentenceBuilder):
    organ_color = {
        'skin': [228, 200, 166],
        'bone': [255, 255, 255],
        'C1': [206, 210, 235],
        'C2': [221, 200, 220],
        'C3': [251, 190, 190],
        'C4': [255, 190, 190],
        'C5': [255, 190, 190],
        'C6': [255, 190, 190],
        'C7': [255, 190, 190],
        'T1': [205, 255, 190],
        'T2': [190, 255, 190],
        'T3': [190, 255, 190],
        'T4': [190, 255, 190],
        'T5': [190, 255, 190],
        'T6': [190, 255, 190],
        'T7': [190, 255, 190],
        'T8': [190, 255, 190],
        'T9': [190, 255, 190],
        'T10': [190, 255, 190],
        'T11': [190, 255, 190],
        'T12': [190, 255, 190],
        'L1': [240, 190, 235],
        'L2': [230, 190, 250],
        'L3': [210, 190, 255],
        'L4': [190, 190, 255],
        'L5': [190, 190, 255],
        'subcutaneous_fat': [255, 255, 0],
        'muscle': [255, 0, 0],
        'torso_fat': [139, 69, 19],
        'chest_organs': [0, 0, 255],
        'belly_organs': [0, 255, 0],
        'brain':[128, 0, 128],
    }
    def __init__(self, root_dir, hu_range=(-1000, 3000)):
        self.hu_range = hu_range
        self.root_dir = root_dir

    def load_npz(self, npz_path):
        data = np.load(npz_path)
        return data['data'], data['info'].item()

    def interpolate_slice(self, slice_data, target_size):
        _, h, w, _ = slice_data.shape
        scale_factor = target_size / max(h, w)
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        slice_data = torch.from_numpy(slice_data).to(torch.float32)
        slice_data = slice_data.permute(0, 3, 1, 2)
        slice_data = F.interpolate(slice_data, size=(new_h, new_w), mode='bilinear', align_corners=False)
        if new_h < target_size or new_w < target_size:
            pad_h = target_size - new_h
            pad_w = target_size - new_w
            slice_data = F.pad(slice_data, (0, pad_w, 0, pad_h), mode='constant', value=-1)
        return slice_data.permute(0, 2, 3, 1).numpy()

    def normalize(self, data, min_val=-1, max_val=1):
        data_min = np.min(data)
        data_max = np.max(data)
        return (max_val - min_val) * (data - data_min) / (data_max - data_min) + min_val

    def to_frames(self, data, axis, start_index, target_size, num_frames=32):
        assert 0 <= start_index < data.shape[axis] - num_frames, f"Invalid start index for axis {axis}"
        
        if axis == 0:
            data = data[start_index:start_index+num_frames,:,:,:]
        elif axis == 1:
            data = data[:,start_index:start_index+num_frames,:,:]
        else:
            data = data[:,:,start_index:start_index+num_frames,:]

        frames = []
        for i in range(num_frames):
            if axis == 0:
                slice_data = data[i:i+1,:,:,:]
            elif axis == 1:
                slice_data = data[:,i:i+1,:,:]
            else:
                slice_data = data[:,:,i:i+1,:]
            slice_data = self.interpolate_slice(slice_data, target_size)
            rgb_data = self.normalize(slice_data[...,:3])
            ct_data = self.normalize(slice_data[...,3])
            smpl_data = self.normalize(slice_data[...,4:])
            frames.append(np.concatenate([rgb_data, ct_data, smpl_data], axis=-1))
        frames = np.stack(frames, axis=0)
        return frames

    def __getitem__(self, index):
        npz_path = os.path.join(self.root_dir, f'{index}.npz')
        data, info = self.load_npz(npz_path)
        age = info['age']
        gender = info['gender']
        axis = 0 # 指定轴
        start_index = data.shape[axis] - 32 # 指定起始位置
        frames = self.to_frames(data, axis=axis, start_index=start_index, target_size=256)
        return frames, age, gender
        
    
        