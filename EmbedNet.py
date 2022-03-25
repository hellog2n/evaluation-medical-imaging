
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn.functional as F
import torch
import torch.nn as nn

import sys
from os import path
import numpy as np
#print(path.dirname(path.dirname(path.abspath(__file__))))
#sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x


class EmbedNetwork():
    def __init__(self, net_name, device, outnc=None, extracting_layer=None, finetuned=False, **path):
        self._net_name = net_name
        self._outnc = outnc
        self._device = device
        self._extracting_layer = extracting_layer
        self.resize_input = True
        self._finetuned = finetuned
        self._path = path
        print(self._path)
        self.loadModel()


    def loadModel(self):
        if 'inceptionV3' in self._net_name:
            from inceptionModel import InceptionV3
            import torchvision.models as models
            from torch.nn.functional import adaptive_avg_pool2d

            dims = 2048
            # 이미지를 담을 input_tensor를 선언한다.
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
            if self._finetuned:
                print('load finetuned')
                assert self._path['finetuned_path'], 'You should insert finetuned_path'
                print('load medical finetuned weights (imagenet + medical)')
                model = InceptionV3(
                    [block_idx], finetuned=True, finetuned_path=self._path['finetuned_path']).to(self._device)
            else:
                print('load imagenet pretrained weights')
                model = InceptionV3([block_idx]).to(self._device)
        self.model = model
        self.model.eval()

    def modelEmbedding(self, dataloader, args):
        if args.embeddingModel == 'inceptionV3':
            return self.generate_inception_embedding(dataloader)
        else:
            assert 0, 'embedding model is not defined or wrong defined'

    def generate_inception_embedding(self, dataloader):
        batch_size = 32
        embeddings = []
        start_idx = 0

        for batch in tqdm(dataloader):
            batch = batch.to(self._device)
            

            with torch.no_grad():
                pred = self.model(batch)[0]
                # If model output is not scalar, apply global spatial average pooling.
                # This happens if you choose a dimensionality not equal 2048.
                if pred.size(2) != 1 or pred.size(3) != 1:
                    pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

            pred = pred.cpu().numpy()
            b, f, a, d = pred.shape
            pred = pred.reshape(b, f)
            embeddings.append(pred)
        print('shape', np.concatenate(embeddings, axis=0).shape)
        return np.concatenate(embeddings, axis=0)
