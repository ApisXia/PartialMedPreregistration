from __future__ import division
import torch
import torch.optim as optim
from torch.nn import functional as F
import os
from torch.utils.tensorboard import SummaryWriter
from glob import glob
import numpy as np
from tqdm import tqdm
import time


class Trainer(object):

    def __init__(self, model, device, train_dataset, test_dataset, exp_name, optimizer='Adam', lr=1e-3):
        self.model = model.to(device)
        self.device = device
        if optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr= lr)
        if optimizer == 'Adadelta':
            self.optimizer = optim.Adadelta(self.model.parameters())
        if optimizer == 'RMSprop':
            self.optimizer = optim.RMSprop(self.model.parameters(), momentum=0.9)

        self.train_dataset = train_dataset
        self.val_dataset = test_dataset
        self.exp_path = os.path.dirname(__file__) + '/experiments/{}/'.format( exp_name)
        self.checkpoint_path = self.exp_path + 'checkpoints/'.format(exp_name)
        if not os.path.exists(self.checkpoint_path):
            print(self.checkpoint_path)
            os.makedirs(self.checkpoint_path)
        self.writer = SummaryWriter(self.exp_path + 'summary'.format(exp_name))
        self.val_min = None

    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.compute_loss(batch)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def compute_loss(self, batch):
        device = self.device

        # ct = batch.get('ct_matrix').to(device)
        ra = batch.get('ra_matrix').to(device)
        loc = batch.get('ra_loc').to(device)

        loc_pred = self.model(ra)

        loss_i = torch.nn.L1Loss(reduction='none')(loc_pred, loc)
        loss = loss_i.sum(-1).mean()

        return loss

    def train_model(self, epochs):
        loss = 0
        train_data_loader = self.train_dataset.get_loader()
        start = self.load_checkpoint()

        for epoch in range(start, epochs):
            sum_loss = 0
            print('Start epoch {}'.format(epoch))

            with tqdm(train_data_loader, smoothing=0.2) as desc:
                for batch in desc:
                    #optimize model
                    loss = self.train_step(batch)
                    # print("Current loss: {}".format(loss / self.train_dataset.num_sample_points))
                    desc.set_postfix({"Current loss": loss})
                    desc.set_description_str("Epoch: {}. Training stage".format(epoch))
                    sum_loss += loss

            self.writer.add_scalar('training loss last batch', loss, epoch)
            self.writer.add_scalar('training loss batch avg', sum_loss / len(train_data_loader), epoch)

            # save model
            if (epoch + 1) % 5 == 0:  # eval model every X min and at start
                self.save_checkpoint(epoch)
                val_loss = self.compute_val_loss()

                if self.val_min is None:
                    self.val_min = val_loss

                if val_loss < self.val_min:
                    self.val_min = val_loss
                    for path in glob(self.exp_path + 'val_min=*'):
                        os.remove(path)
                    np.save(self.exp_path + 'val_min={}'.format(epoch), [epoch, val_loss])

                self.writer.add_scalar('val loss batch avg', val_loss, epoch)
                print('Epoch: {}. Val loss={}'.format(epoch, val_loss))

    def save_checkpoint(self, epoch):
        path = self.checkpoint_path + 'checkpoint_{}.tar'.format(epoch)
        if not os.path.exists(path):
            torch.save({ #'state': torch.cuda.get_rng_state_all(),
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict()}, path)

    def load_checkpoint(self):
        checkpoints = glob(self.checkpoint_path+'/*')
        if len(checkpoints) == 0:
            print('No checkpoints found at {}'.format(self.checkpoint_path))
            return 0

        checkpoints = [os.path.splitext(os.path.basename(path))[0].split('_')[-1] for path in checkpoints]
        checkpoints = np.array(checkpoints, dtype=float)
        checkpoints = np.sort(checkpoints)
        # path = self.checkpoint_path + 'checkpoint_{}h:{}m:{}s_{}.tar'.format(*[*convertSecs(checkpoints[-1]),checkpoints[-1]])
        path = self.checkpoint_path + 'checkpoint_{}.tar'.format(int(checkpoints[-1]))

        print('Loaded checkpoint from: {}'.format(path))
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        # training_time = checkpoint['training_time']
        # torch.cuda.set_rng_state_all(checkpoint['state']) # batch order is restored. unfortunately doesn't work like that.
        return epoch

    def compute_val_loss(self):
        self.model.eval()

        sum_val_loss = 0
        num_batches = 15
        for _ in range(num_batches):
            try:
                val_batch = self.val_data_iterator.next()
            except:
                self.val_data_iterator = self.val_dataset.get_loader().__iter__()
                val_batch = self.val_data_iterator.next()

            sum_val_loss += self.compute_loss(val_batch).item()

        return sum_val_loss / num_batches
