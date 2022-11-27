import glob
import math
import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from torch import nn
from torch.utils.data import DataLoader, Dataset

"""
Class storing NFE GAN variables and methods
"""


class NFEModel():
    """
    Load models and init image manipulation svm
    """

    def __init__(self, root):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.G = torch.load(os.path.join(root, "data", "Generator_v2_150.pth"), map_location=self.device)
        self.E = torch.load(os.path.join(root, "data", "Encoder.pth"), map_location=self.device)

        self.properties = ["hair", "hair_length", "eyes"]
        self.feature2svm = {}
        self.class2value = {}
        self.dataroot = os.path.join(root, "data")
        path = os.path.join(self.dataroot, "attributes.csv")
        attributes = pd.read_csv(path)
        self.property_to_options = {}
        for i in self.properties:
            filename = f'svm_{i}.sav'
            svm = pickle.load(open(os.path.join(self.dataroot, filename), 'rb'))
            self.feature2svm[i] = svm
            y = attributes[~attributes[i].isna()][i]
            self.property_to_options[i] = y.unique()
            label_encoder = LabelEncoder()
            label_encoder.fit(y)
            self.class2value[i] = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

    """
    Apply generator on latent vector z.
    """

    def apply_generator(self, z):
        step = int(math.log(64, 2)) - 2
        with torch.no_grad():
            z = z.to(self.device)
            img = self.G(z, step=step)[0]
        return img

    """
    Generate image from latent vector z and save image by imgpath.
    """

    def generate_and_save_image(self, z, imgpath):
        img = self.apply_generator(z)
        imgdata = torch.clip(img, 0, 1).permute([1, 2, 0]).detach().cpu().numpy()
        plt.imsave(imgpath, imgdata)

    """
    Generate random latent vector z, get image from generator and save by imgpath.
    """

    def generate_and_save_random_image(self, imgpath):
        z = torch.randn((1, 512))
        self.generate_and_save_image(z, imgpath)
        return z

    """
    Perform GAN inversion model training.
    """

    def gan_inversion(self, imgpath):
        # transform image
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        img = Image.open(imgpath).convert("RGB")
        img = transform(img).unsqueeze(0)

        z_dim = 512
        step = int(math.log(z_dim, 2)) - 2
        img = img.to(self.device)

        with torch.no_grad():
            z = self.E(img.to(self.device))
        z = torch.clip(z, 0, 1)
        z.requires_grad_(True)
        z.retain_grad()
        lr = 1e-2
        optimizer = torch.optim.Adam([z], lr=lr)

        criterion = torch.nn.MSELoss(reduction="sum")

        losses = []
        epochs = 1000
        f = transforms.functional.gaussian_blur
        for i in range(epochs):
            optimizer.zero_grad()

            out = self.G(z, step=step)

            loss = criterion(f(out, 3, 11.0), f(img, 3, 11.0)) + 20.0 * torch.norm(z, p=2.0)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

            if i % 200 == 0:
                print("iter {:04d}: y_error = {:03g}".format(i,
                                                             loss.item()))
        return z

    def get_conditional_boundary(self, boundary, preserved_features=None):
        """
        make the hyperplane orthogonal to hyperplanes of specified features
        :param boundary: hyperplane of the feature to be changed
        :param preserved_features: list of feature names to be preserved
        :return: modified separating hyperplane
        """
        if preserved_features is None:
            return boundary / np.linalg.norm(boundary)
        cond = []
        for feature in preserved_features:
            cond.append(self.feature2svm[feature].coef_)
        cond = np.vstack(cond)
        A = np.matmul(cond, cond.T)
        B = np.matmul(cond, boundary.T)
        x = np.linalg.solve(A, B)
        new = boundary - (np.matmul(x.T, cond))
        return new / np.linalg.norm(new)

    def manipulate(self, z, feature, value, preserved_features=None, start=-3.0, end=3.0, steps=7, z_dim=512):
        """
        Manipulate the image in the desired attribute
        :param z: latent code to be manipulated
        :param feature: feature to be changed
        :param value: desired target value of the feature
        :param preserved_features: features to be preserved
        :param start: The distance to the boundary where the manipulation starts
        :param end: The distance to the boundary where the manipulation ends
        :param steps: Number of manipulation steps between start and end
        :param z_dim: dimensionality of latent vector
        :return: list of manipulated latent vectors
        """
        if not os.path.exists(os.path.join(self.dataroot, f'svm_{feature}.sav')):
            raise AssertionError('No svm found')
            vectors = np.loadtxt(os.path.join(root, "vectors.csv"), delimiter=",")
            svm = get_boundary(vectors, attributes, feature)
        else:
            svm = self.feature2svm[feature]
        if len(z.shape) == 1:
            z = z.reshape((-1, z_dim))
        init_value = svm.predict(z)[0]
        value = self.class2value[feature][value]
        if value == init_value:
            value = random.choice([i for i in range(len(self.class2value[feature])) if i != init_value])
        boundary_idx = 0
        boundary_idx += sum([len(self.class2value[feature]) - i - 1 for i in range(min(init_value, value))])
        boundary_idx += abs(init_value - value) - 1
        boundary = self.get_conditional_boundary(svm.coef_[boundary_idx], preserved_features)
        linspace = np.linspace(start, end, steps) - z.dot(boundary.T)
        return z + linspace.reshape(steps, 1) * boundary.reshape(1, -1)

    """
    Return array of pictures by filepaths
    """
    def picture_array(self, filepaths):
        pics = []
        for filepath in filepaths:
            pics.append(open(filepath, 'rb'))
        return pics

    """
    Image manipulation wrapper function.
    """
    def change_image(self, path_prefix, z, feature='hair', value='green', preserved_features=None):
        img_size = 64
        z_s = self.manipulate(z, feature, value, preserved_features, start=-5.0, end=5.0)
        step = int(math.log(img_size, 2)) - 2
        pics = []
        # save generated images and return to the user
        filepath = '{}-fig{}.png'
        for i, z in enumerate(z_s):
            z = torch.tensor(z).to(self.device)
            out = self.G(z.view(1, -1).float(), step=step)
            img = out[0].permute([1, 2, 0]).detach().cpu().numpy()
            img = np.clip(img, 0, 1)
            plt.imsave(filepath.format(path_prefix, i), img)
            pics.append(open(filepath.format(path_prefix, i), 'rb'))
        return pics
