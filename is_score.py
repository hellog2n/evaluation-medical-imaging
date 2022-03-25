import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
from PIL import Image

import numpy as np
from scipy.stats import entropy
import torchvision.models as models


def inception_score(imgs, device=None, batch_size=32, resize=False, splits=1, finetuned=False, **kwargs):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = 6080
    dataloader = imgs
    assert batch_size > 0
    assert N > batch_size

    print('load model weight')
    # Load inception model
    inception_model = models.inception_v3(pretrained=True).to(device)

    if finetuned:
        print('load finetuned')
        assert kwargs['finetuned_path'], 'You should insert finetuned_path'
        finetuned_path = kwargs['finetuned_path']
        inception_model.fc = nn.Linear(2048, 2).to(device)
        inception_model.AuxLogits.fc = nn.Linear(768, 2).to(device)
        inception_model.load_state_dict(
            torch.load(finetuned_path), strict=True)

    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False)

    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    if not finetuned:
        preds = np.zeros((N, 1000))
    else:
        preds = np.zeros((N, 2))

    for i, batch in enumerate(dataloader, 0):

        batch = batch.to(device)
        batchv = Variable(batch).to(device)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)
