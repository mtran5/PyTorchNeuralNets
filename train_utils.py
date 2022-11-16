import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from skimage.draw import random_shapes
import torchvision.transforms as transforms
from tqdm import tqdm

class Trainer():
    def __init__(self, model, dataloader, optimizier, loss_fn, epochs, device, dtype=torch.float32):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizier
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.device = device
        self.dtype = dtype

    def train(self, save=False, savemode="val", savepath=None):
        self.model.train()
        self.model.to(self.device)
        for e in range(self.epochs):
            print("Epoch {0} out of {1}".format(e+1, self.epochs))
            print("_"*10)
            epoch_loss = 0.0

            for t, (x, y) in enumerate(tqdm(self.dataloader)):
                x = x.to(self.device, dtype=self.dtype)
                scores = self.model(x)
                y = y.to(self.device).type_as(scores)
                loss = self.loss_fn(scores, y)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                with torch.no_grad():
                    epoch_loss += loss.item()
            print(f"epoch loss: {epoch_loss}")
        
        # save model
        if save:
            torch.save(self.model.state_dict(), savepath)

    def eval(self, validation_loader):
        self.model.eval()
        self.model.to(self.device)
        scores_array = np.array([])
        with torch.no_grad():
            for t, (x, y) in enumerate(tqdm(validation_loader)):
                x = x.to(self.device, dtype=self.dtype)
                scores = self.model(x)
                scores_array += scores.flatten().detach().cpu().numpy()
        return scores_array


