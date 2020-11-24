from dcgan_dataset import Dataset, collate
from torch.utils import data
import torch.optim as optim
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import joblib
import multiprocessing
import concurrent.futures
from dtwalign import dtw
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import scipy.interpolate as interp
from scipy.signal import butter, lfilter
import collections
from functools import partial
from preprocessing import collect_files
import yaml
from easydict import EasyDict as edict
import utils  # this creates a partition if one doesn't exist, but use existing one for consistency
import shutil
import json

config = edict(yaml.load(open('config.yml'), Loader=yaml.SafeLoader))

"""Need to pair the corresponding feature files at the beginning"""

"""Experiment settings"""
ALIGN = False  # this just runs dtw, we are using dtw-aligned data for both vc methods
TRAIN = True
EVAL = False
PLOT_ACTIVATIONS = False
exp_name = 'PARTITION_2_trial_1_dcgan_vc_model_training'
GLOBAL_PARTITION = 2

"""Create directories"""
if not os.path.isdir('exps'):
    os.mkdir('exps')
exp_root = os.path.join('exps', exp_name)
if not os.path.isdir(exp_root):
    os.mkdir(exp_root)

TRANSFORMER = False
LOAD_MODEL = False

activations = collections.defaultdict(list)  # collect activations in attention from first layer

# Signal Processing Hyperparameters
FRAME_LENGTH = config.data.fftl
HOP_LENGTH = config.data.hop_length  # 160
SAMPLE_RATE = config.data.sr
NFFT = config.data.fftl  # MCEP and Mel_log_spect features need to have the same frames
NUM_MELS = config.data.num_mels
TOP_DB = config.data.top_db

# Neural Network Hyperparameters
NUM_EPOCHS = 6000  # this is kind of a dummy variable, we go by TOTAL_ITERATIONS
TOTAL_ITERATIONS = 10003

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.num_conv_channels = 8  # this is 8 in their paper, trying more filters
        if config.dcgan.use_batchnorm:
            self.main = nn.Sequential(
                # input is spectral features
                # Conv1
                nn.Conv2d(in_channels=1, out_channels=self.num_conv_channels, kernel_size=(5, 5),
                          stride=(1, 1), padding=2, padding_mode='zeros'),
                nn.BatchNorm2d(self.num_conv_channels),
                #nn.ReLU(True),
                nn.LeakyReLU(True),
                # Conv2
                nn.Conv2d(in_channels=self.num_conv_channels, out_channels=self.num_conv_channels, kernel_size=(5, 5),
                          stride=(1, 1), padding=2, padding_mode='zeros'),
                nn.BatchNorm2d(self.num_conv_channels),
                #nn.ReLU(True),
                nn.LeakyReLU(True),
                # Conv3
                nn.Conv2d(in_channels=self.num_conv_channels, out_channels=1, kernel_size=(5, 5),
                          stride=(1, 1), padding=2, padding_mode='zeros'),
                # nn.ReLU(True),  # you weren't getting any negative numbers!
            )
        else:
            self.main = nn.Sequential(
                # input is spectral features
                # Conv1
                nn.Conv2d(in_channels=1, out_channels=self.num_conv_channels, kernel_size=(5, 5),
                          stride=(1, 1), padding=2, padding_mode='zeros'),
                # nn.ReLU(True),
                nn.LeakyReLU(True),
                # Conv2
                nn.Conv2d(in_channels=self.num_conv_channels, out_channels=self.num_conv_channels, kernel_size=(5, 5),
                          stride=(1, 1), padding=2, padding_mode='zeros'),
                # nn.ReLU(True),
                nn.LeakyReLU(True),
                # Conv3
                nn.Conv2d(in_channels=self.num_conv_channels, out_channels=1, kernel_size=(5, 5),
                          stride=(1, 1), padding=2, padding_mode='zeros'),
                # nn.ReLU(True),
            )


    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, nfreqs, adapt_size):
        super(Discriminator, self).__init__()
        self.nfreqs = nfreqs
        self.adapt_size = adapt_size
        if config.dcgan.use_batchnorm:
            self.conv_layers = nn.Sequential(
                # input is spectral features
                # Conv1
                nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(5, 5),
                          stride=(2, 2), padding=2, padding_mode='zeros'),
                # nn.ReLU(True),
                nn.LeakyReLU(True),
                # Conv2
                nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(5, 5),
                          stride=(2, 2), padding=2, padding_mode='zeros'),
                nn.BatchNorm2d(16),
                # nn.ReLU(True),
                nn.LeakyReLU(True),
                # Conv3
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5),
                          stride=(2, 2), padding=2, padding_mode='zeros'),
                nn.BatchNorm2d(32),
                # nn.ReLU(True),
                nn.LeakyReLU(True),
                # Conv4
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5),
                          stride=(2, 2), padding=2, padding_mode='zeros'),
                nn.BatchNorm2d(64),
                # nn.ReLU(True),
                nn.LeakyReLU(True),
            )
        else:
            self.conv_layers = nn.Sequential(
                # input is spectral features
                # Conv1
                nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(5, 5),
                          stride=(2, 2), padding=2, padding_mode='zeros'),
                # nn.ReLU(True),
                nn.LeakyReLU(True),
                # Conv2
                nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(5, 5),
                          stride=(2, 2), padding=2, padding_mode='zeros'),
                # nn.ReLU(True),
                nn.LeakyReLU(True),
                # Conv3
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5),
                          stride=(2, 2), padding=2, padding_mode='zeros'),
                # nn.ReLU(True),
                nn.LeakyReLU(True),
                # Conv4
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5),
                          stride=(2, 2), padding=2, padding_mode='zeros'),
                # nn.ReLU(True),
                nn.LeakyReLU(True),
            )

        self.avg_pooling = nn.AdaptiveAvgPool2d(output_size=(self.nfreqs, self.adapt_size))
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(in_features=64*self.nfreqs*self.adapt_size,
                            out_features=1)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.avg_pooling(x)
        x = self.flatten(x)
        x = self.fc(x)
        return F.sigmoid(x)

def positional_encodings(nframes, d_model):
    times = np.arange(nframes, dtype=np.float)
    frequencies = np.power(10000.0, -2 * np.arange(d_model / 2) / d_model)
    phases = torch.tensor(data=np.outer(times, frequencies), dtype=torch.float)
    return (torch.cat((torch.sin(phases), torch.cos(phases)), dim=1))

def interpolate_resample(array, desired_length):
    # link: https://stackoverflow.com/questions/38064697/interpolating-a-numpy-array-to-fit-another-array
    arr1_interp = interp.interp1d(np.arange(array.shape[0]), array, kind='linear')
    new_array = arr1_interp(np.linspace(0, array.shape[0] - 1, desired_length))
    return new_array

def stretch(target, y):
    length = target.shape[0]
    y = y.T
    new_y = []
    for i in range(y.shape[0]):
        new_y.append(interpolate_resample(y[i], length))
    new_y = np.asarray(new_y)
    y = new_y.T
    return y

def butter_lowpass(cutOff, fs, order=5):
    nyq = 0.5 * fs
    normalCutoff = cutOff / nyq
    b, a = butter(order, normalCutoff, btype='low', analog=False)  # analog=False critical for moving average filter
    return b, a

def butter_lowpass_filter(data, cutOff, fs, order=4):
    b, a = butter_lowpass(cutOff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def save_activation(name, mod, inp, out):
    # out[1] is the attention weights
    act = np.squeeze(out[1].detach().cpu().numpy())
    activations[name].append(act)

def get_conversion_pairs():
    pairs = []
    for source in config.data.conversion_source_speakers:
        for target in config.data.conversion_target_speakers:
            pairs.append({'source': source, 'target': target})
    return pairs

class Solver(object):
    """Solver"""

    def __init__(self, pair_dir, normal_speaker, dys_speaker):
        """Initialize configurations."""

        # Speakers
        self.normal_speaker = normal_speaker
        self.dys_speaker = dys_speaker

        # Training configurations.
        self.g_lr = config.dcgan.g_lr
        self.d_lr = config.dcgan.d_lr
        self.torch_type = torch.float32
        """Initialize BCELoss function"""
        self.criterion = nn.BCELoss()

        """Establish convention for real and fake labels during training"""
        self.real_label = 1
        self.fake_label = 0

        # Miscellaneous.
        self.use_tensorboard = True
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:{}'.format(0) if self.use_cuda else 'cpu')

        # Directories.
        self.root_dir = pair_dir
        self.log_dir = os.path.join(self.root_dir, 'logs')
        self.model_save_dir = os.path.join(self.root_dir, 'models')
        self.image_save_dir = os.path.join(self.root_dir, 'train_images')
        self.converted_audio_dir = os.path.join(self.root_dir, 'converted_audio')
        if not os.path.isdir(self.log_dir):
            os.mkdir(self.log_dir)
        if not os.path.isdir(self.model_save_dir):
            os.mkdir(self.model_save_dir)
        if not os.path.isdir(self.image_save_dir):
            os.mkdir(self.image_save_dir)
        if not os.path.isdir(self.converted_audio_dir):
            os.mkdir(self.converted_audio_dir)

        # Partition file
        self.partition = 'full_filenames_data_partition.pkl'
        partition_copy_path = os.path.join(self.root_dir, 'full_filenames_data_partition.pkl')
        shutil.copy(src=self.partition, dst=partition_copy_path)

        # Step size.
        self.log_step = 100
        self.model_save_step = 5000

        # Build the model
        self.build_model()
        if EVAL or LOAD_MODEL:
            self.restore_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        """Create a generator and a discriminator."""
        self.G = Generator()
        self.D = Discriminator(nfreqs=NUM_MELS, adapt_size=config.dcgan.adapt_size)
        self.optimizerD = optim.Adam(self.D.parameters(), lr=self.d_lr, betas=(config.dcgan.beta1, 0.999))
        self.optimizerG = optim.Adam(self.G.parameters(), lr=self.g_lr, betas=(config.dcgan.beta1, 0.999))
        self.print_network(self.G, 'G')
        self.G.to(self.device)
        self.print_network(self.D, 'D')
        self.D.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def print_optimizer(self, opt, name):
        print(opt)
        print(name)

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def _load(self, checkpoint_path):
        if self.use_cuda:
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path,
                                    map_location=lambda storage, loc: storage)
        return checkpoint

    def restore_model(self, path=None):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models... ')
        if path != None:
            G_path = path
        else:
            G_path = os.path.join(self.model_save_dir, '10000-G.ckpt')
        g_checkpoint = self._load(G_path)
        self.G.load_state_dict(g_checkpoint['model'])
        self.optimizerG.load_state_dict(g_checkpoint['optimizer'])
        self.g_lr = self.optimizerG.param_groups[0]['lr']

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()

    def save_images(self, i, output, target, input):
        output = np.squeeze(output.detach().cpu().numpy())
        target = np.squeeze(target.detach().cpu().numpy())
        input = np.squeeze(input.detach().cpu().numpy())
        plt.subplot(311)
        plt.imshow(output.T)
        plt.subplot(312)
        plt.imshow(target.T)
        plt.subplot(313)
        plt.imshow(input.T)
        plt.savefig(os.path.join(self.image_save_dir,  str(i) + '.png'))
        plt.close()

    def get_train_val(self, pair_dir, normal_speaker, dys_speaker):
        """WE MAKE A PAIR TRAIN/VAL PARTITION HERE AND SAVE TO EXPERIMENT
                We shouldn't need this partition for anything else but just save for records"""
        # wordlist = joblib.load('wordlist.pkl')  # this is to save time so we don't load 'wordlist.pkl' for every file
        # dictionary = joblib.load('dict.pkl')  # ditto to above
        files = joblib.load(self.partition)
        files = files['partition_' + str(GLOBAL_PARTITION)]['seen']['filenames_only']
        # train_files = files['train']
        # val_files = files['val']
        # test_files = files['test']

        # intersection = set(train_files).intersection(set(test_files))

        # """Convert test_files to dict for fast searching (hashable, try except block)"""
        # test_dict = {}
        # for file in test_files:
        #     test_dict[file] = ''
        # """Convert val_files to dict for fast searching (hashable, try except block)"""
        # val_dict = {}
        # for file in val_files:
        #     val_dict[file] = ''
        # """Convert train_files to dict for fast searching (hashable, try except block)"""
        # train_dict = {}
        # for file in train_files:
        #     train_dict[file] = ''

        """Collect all paired files in the pair_dir"""
        all_files = collect_files(os.path.join(config.directories.dtw, pair_dir.split('/')[-1]))
        normal_seen_files = (files['normal']['train'].copy())
        normal_seen_files.extend(files['normal']['val'].copy())
        normal_seen_files.extend(files['normal']['test'].copy())

        dys_seen_files = (files['dys']['train'].copy())
        dys_seen_files.extend(files['dys']['val'].copy())
        dys_seen_files.extend(files['dys']['test'].copy())
        usable_files = []
        for pair in all_files:
            """Construct the original paths of BOTH speakers and make sure in seen set"""
            file_end = (pair.split('/')[-1]).replace(normal_speaker + '_to_' + dys_speaker, '')
            normal_original_path = os.path.join(config.directories.features, normal_speaker + file_end)
            dys_original_path = os.path.join(config.directories.features, dys_speaker + file_end)

            if normal_original_path in normal_seen_files and dys_original_path in dys_seen_files:
                usable_files.append(pair)

            # if normal_original_path in test_dict or dys_original_path in test_dict:
            #     train_usable = False
            # if normal_original_path in val_dict or dys_original_path in val_dict:
            #     train_usable = False
            # if train_usable:
            #     usable_train_files.append(pair)
            # val_usable = True
            # if normal_original_path in test_dict or dys_original_path in test_dict:
            #     val_usable = False
            # if normal_original_path in train_dict or dys_original_path in train_dict:
            #     val_usable = False
            # if val_usable:
            #     usable_val_files.append(pair)

        """Make the pair-wise partition after testing this code"""
        size = len(usable_files)
        train_size = round(0.98 * size)
        train_files = usable_files[0:train_size]
        val_files = usable_files[train_size:]

        return train_files, val_files

    def get_batch_transformer(self, file):
        name = file.split('/')[-1]
        data = joblib.load(file)
        src = data['normal']
        tgt = data['dys']

        """This is an artifact from when we weren't doing dtw for this but rather just stretching"""
        # need to stretch them to be the same length
        # src = stretch(tgt, src)

        # plt.subplot(211)
        # plt.imshow(tgt.T)
        # plt.subplot(212)
        # plt.imshow(src.T)
        # plt.show()

        src_len = src.shape[0]
        tgt_len = tgt.shape[0]
        src = torch.from_numpy(src)
        tgt = torch.from_numpy(tgt)

        # add the positional encodings
        src_position_enc = positional_encodings(nframes=src_len, d_model=self.G.d_model)
        tgt_position_enc = positional_encodings(nframes=tgt_len, d_model=self.G.d_model)

        src = src + 1e-6*src_position_enc
        tgt = tgt + 1e-6*tgt_position_enc

        # adjust shapes for input into transformer
        src = torch.unsqueeze(src, dim=1)
        tgt = torch.unsqueeze(tgt, dim=1)

        src = self.to_gpu(src)
        tgt = self.to_gpu(tgt)

        return src, tgt

    def get_batch_transformer_eval(self, file):
        name = file.split('/')[-1]
        try:
            src = joblib.load(os.path.join('features', self.normal_speaker + '_' + name + '.pkl'))
        except:
            return None

        src_len = src.shape[0]
        stretch_dummy = np.zeros(shape=(int(src_len*1.4), src.shape[1]))

        # need to stretch src to be 1.4 times its original length
        src = stretch(stretch_dummy, src)

        # plt.subplot(211)
        # plt.imshow(tgt.T)
        # plt.subplot(212)
        # plt.imshow(src.T)
        # plt.show()

        src_len = src.shape[0]
        src = torch.from_numpy(src)

        # add the positional encodings
        src_position_enc = positional_encodings(nframes=src_len, d_model=self.G.d_model)

        src = src + 1e-6 * src_position_enc

        # adjust shapes for input into transformer
        src = torch.unsqueeze(src, dim=1)

        src = self.to_gpu(src)

        return src

    def val_loss_transformer(self, val):
        random.shuffle(val)
        total_loss = 0
        self.G = self.G.eval()
        counter = 0
        for i, file in enumerate(val):
            if i < 5:
                src, tgt = self.get_batch_transformer(file)
                try:  # not all speakers have all utterances
                    input_encoder_outputs = self.G(src)
                    loss = F.mse_loss(input_encoder_outputs, tgt, reduction='sum')
                    total_loss += loss.item()
                    counter += 1
                except:
                    print("Utterance not in dysarthric speaker set...")
        self.G = self.G.train()
        return total_loss/counter  # divide by counter in case we miss an utterance (see above)

    def train_dcgan(self, pair_dir, normal_speaker, dys_speaker):
        """We are going to let the batch size be 1 for the transformer so we don't pad stuff"""
        iterations = 0
        """YOU NEED TO MAKE SURE THERE ARE NO FILES FROM THE TEST PARTITION
        IN HERE. YOU MUST TAKE EXTRA CARE SINCE YOU HAVE PAIRS NOW, SO YOU MUST CHECK BOTH FILES
        FROM THE PAIR AND IF EITHER OCCURS IN TEST SET YOU CAN'T USE IT"""

        """Also make sure to save partition to the experiment directory"""
        """Already did this in initialization, doing it for each pair since they take several hours"""
        train, val = self.get_train_val(pair_dir, normal_speaker, dys_speaker)
        """Save the train/val partition"""
        joblib.dump({'train': train, 'val': val}, os.path.join(self.root_dir, 'pair_partition.pkl'))
        len_train_val = {'train_files': len(train), 'val_files': len(val)}

        with open(os.path.join(self.root_dir, 'len_train_val.txt'), 'w') as file:
            file.write(json.dumps(len_train_val))  # use `json.loads` to do the reverse
        print('***********************************************')
        print(len(train))
        print(len(val))
        print('***********************************************')
        """Make dataloader"""
        train_data = Dataset({'pairs': train, "mode": 'train'})
        train_gen = data.DataLoader(train_data, batch_size=config.dcgan.batch_size, shuffle=True, collate_fn=collate, drop_last=True)
        for epoch in range(NUM_EPOCHS):
            if iterations < TOTAL_ITERATIONS:
                random.shuffle(train)
                for batch_number, batch in enumerate(train_gen):
                    if iterations < TOTAL_ITERATIONS:
                        """"""
                        # normal_numpy = batch['normal'].detach().cpu().numpy()
                        # normal_numpy = np.squeeze(normal_numpy)
                        # example = normal_numpy[0]
                        # plt.imshow(example.T)
                        # plt.show()
                        try:  # some speakers don't have all the utterances
                            normal = batch['normal']
                            dys = batch['dys']
                            """Do a training pass"""
                            ############################
                            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                            #     Only update every d_schedule times
                            ###########################
                            if iterations % config.dcgan.d_schedule == 0:
                                """Train with all-real batch"""
                                self.D.zero_grad()
                                label = torch.full((config.dcgan.batch_size,), self.real_label, device=self.device)
                                """Forward pass real batch through D"""
                                output = self.D(dys).view(-1)
                                """Calculate loss on all-real batch"""
                                errD_real = self.criterion(output, label)
                                """Calculate gradients for D in backward pass"""
                                errD_real.backward()
                                D_x = output.mean().item()

                                """Train with all-fake batch"""
                                """Generate fake spectral features batch with G"""
                                fake = self.G(normal)
                                label.fill_(self.fake_label)
                                """Classify all fake batch with D"""
                                output = self.D(fake.detach()).view(
                                    -1)  # detach so the gradients through G aren't kept (they aren't used to update D)
                                """Calculate D's loss on the all-fake batch"""
                                errD_fake = self.criterion(output, label)
                                """Calculate the gradients for this batch"""
                                errD_fake.backward()
                                D_G_z1 = output.mean().item()
                                """Add the gradients from the all-real and all-fake batches"""
                                errD = errD_real + errD_fake
                                """Update D"""
                                self.optimizerD.step()
                                # print("Updated D...")

                            ############################
                            # (2) Update G network: maximize log(D(G(z)))
                            ###########################
                            if iterations % config.dcgan.d_schedule != 0:
                                fake = self.G(normal)  # if we don't calculate from D update step we must do it here
                            self.G.zero_grad()
                            label.fill_(self.real_label)  # fake labels are real for generator cost
                            """Since we just updated D, perform another forward pass of all-fake batch through D"""
                            output = self.D(fake).view(-1)
                            """Calculate G's loss based on this output"""
                            errG = self.criterion(output, label)
                            """Calculate gradients for G"""
                            errG.backward()
                            D_G_z2 = output.mean().item()
                            """Update G"""
                            self.optimizerG.step()
                            # print("Updated G...")

                            if iterations % self.log_step == 0:
                                print(iterations)
                                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                                      % (epoch, NUM_EPOCHS, iterations, len(train_gen),
                                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                                if self.use_tensorboard:
                                    self.logger.scalar_summary('D-loss', errD.item(), iterations)
                                    self.logger.scalar_summary('G-loss', errG.item(), iterations)

                            """Save output spectrograms and dys spectrograms"""
                            if iterations % 200 == 0:
                                fake_np = np.squeeze(fake[0].detach().cpu().numpy())  # choose first element in batch
                                dys_np = np.squeeze(dys[0].detach().cpu().numpy())  # choose first element in batch
                                plt.subplot(211)
                                plt.imshow(fake_np[:, 2:].T)
                                plt.subplot(212)
                                plt.imshow(dys_np[:, 2:].T)
                                plt.savefig(os.path.join(self.image_save_dir, str(iterations) + '.png'))
                                plt.close()

                            # Save model checkpoints.
                            if iterations % self.model_save_step == 0:
                                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(iterations))
                                torch.save({'model': self.G.state_dict(),
                                            'optimizer': self.optimizerG.state_dict()}, G_path)
                                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(epoch))
                                torch.save({'model': self.D.state_dict(),
                                            'optimizer': self.optimizerD.state_dict()}, D_path)
                                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

                            iterations += 1
                        except:
                            print("Utterance not in dysarthric speaker set...")

    def to_gpu(self, tensor):
        tensor = tensor.to(self.torch_type)
        tensor = tensor.to(self.device)
        return tensor

    def eval_transformer(self):
        train, val = self.get_train_val()
        # setup hooks to see activations in attention
        # https://gist.github.com/Tushar-N/680633ec18f5cb4b47933da7d10902af
        # self.hooks = {}
        # for name, module in self.G.named_modules():
        #     self.hooks[name] = module.register_forward_hook(save_activation)

        self.hooks = {}
        self.all_module_names = []
        # I was confused why we don't get attention weights for all heads, but the attention weights
        # are actually averaged in the F.multi_head_attention_forward method and there's no way to change it
        # https://discuss.pytorch.org/t/getting-nn-multiheadattention-attention-weights-for-each-head/72195
        for name, m in self.G.named_modules():
            self.all_module_names.append(type(m))
            if type(m) == nn.modules.activation.MultiheadAttention:
                if 'input' in name:
                    # partial to assign the layer name to each hook
                    self.hooks[name] = m.register_forward_hook(partial(save_activation, name))
        counter = 0
        for i, file in tqdm(enumerate(val)):
            try:
                src = self.get_batch_transformer_eval(file)
                self.G = self.G.eval()
                input_encoder_outputs = self.G(src)
                encoder_spect = np.squeeze(input_encoder_outputs.detach().cpu().numpy())
                # save the converted spectrograms, we'll synthesize them separately
                utt_ID = file.split('/')[-1]
                path__ = os.path.join(self.converted_audio_dir, self.normal_speaker + '_to_' + self.dys_speaker + '_' + utt_ID + '.pkl')
                joblib.dump(encoder_spect, path__)

                src_spect = np.squeeze(src.detach().cpu().numpy())

                # activation plotting
                if PLOT_ACTIVATIONS:
                    self.activations_folder = os.path.join(self.root_dir, 'activations')
                    if not os.path.isdir(self.activations_folder):
                        os.mkdir(self.activations_folder)
                    for layer, value in activations.items():
                        if '0' in layer:
                            attention = value[counter]
                            att_0 = attention[0]
                            att_0_sum = np.sum(att_0)
                            attention = np.flip(attention, axis=1)
                            fig, (ax, ax2) = plt.subplots(nrows=2)
                            ax.imshow(np.squeeze(attention), aspect="auto")
                            ax2.imshow(np.squeeze(src_spect.T), aspect="auto")
                            plt.savefig(os.path.join(self.activations_folder, utt_ID + '_' + layer + '.png'))
                            plt.close()
                counter += 1
            except:
                print("Utterance not in speaker set...")

def dtw_multiprocess(multiprocess_datum):
    pair = multiprocess_datum[0]
    dtw_dir = multiprocess_datum[1]
    normal_speaker = multiprocess_datum[2]
    dys_speaker = multiprocess_datum[3]
    """Need to find the path with mcep features, then apply the path to mel log features (features)"""
    """Also need to make this multiprocessing"""
    mcep_normal_path = pair['normal'].replace(config.directories.features, config.directories.mcep)
    mcep_dys_path = pair['dys'].replace(config.directories.features, config.directories.mcep)

    normal_spec = joblib.load(pair['normal'])
    dys_spec = joblib.load(pair['dys'])
    normal_mcep = (joblib.load(mcep_normal_path))['mcep']
    dys_mcep = (joblib.load(mcep_dys_path))['mcep']
    try:
        res = dtw(normal_mcep, dys_mcep, step_pattern="symmetricP2", open_begin=False, open_end=False)
        normal_path = res.path[:, 0]
        dys_path = res.path[:, 1]

        warped_normal_spect = normal_spec[normal_path]
        warped_dys_spect = dys_spec[dys_path]

        """Name the file"""
        utterance_name = ((pair['normal'].split('/')[-1]).replace(normal_speaker, ''))[1:]
        dump_path = os.path.join(dtw_dir, normal_speaker + '_to_' + dys_speaker + '_' + utterance_name)
        data_to_dump = {'normal': warped_normal_spect, 'dys': warped_dys_spect}
        joblib.dump(data_to_dump, dump_path)
    except:
        print("No alignment found...")

class Process(object):
    def __init__(self):
        self.sr = config.data.sr

    def collect_files(self, directory_list):
        self.all_files = []
        for directory in directory_list:
            for path, subdirs, files in tqdm(os.walk(directory)):
                for name in files:
                    filename = os.path.join(path, name)
                    self.all_files.append(filename)

    def pair_utterances(self):
        self.file_pairs = {}
        # parse the text
        for file in self.all_files:
            name = file.split('/')[-1][:-4]
            name_pieces = name.split('_')
            utterance = name_pieces[1] + '_' + name_pieces[2] + '_' + name_pieces[3]
            if utterance not in self.file_pairs:
                self.file_pairs[utterance] = [file]
            else:
                self.file_pairs[utterance].append(file)

    def get_dtw(self, dtw_dir, normal_speaker, dys_speaker):
        """
        Need to collect all utterances from train portion of partition from both speakers
        Then we find the corresponding utterances and pair them
        Then we dynamically-time warp them and save the features
        """
        feature_files = collect_files(config.directories.features)
        normal_files = []
        dys_files = []
        wordlist = joblib.load('wordlist.pkl')  # this is to save time so we don't load 'wordlist.pkl' for every file
        dictionary = joblib.load('dict.pkl')    # ditto to above
        for file in feature_files:
            metadata = utils.get_file_metadata(file, wordlist=wordlist, dictionary=dictionary)
            if normal_speaker == metadata['speaker']:
                normal_files.append(file)
            elif dys_speaker == metadata['speaker']:
                dys_files.append(file)

        pairs = []
        """Let's find the file pairs"""
        for normal_file in normal_files:
            dys_file = normal_file.replace(normal_speaker, dys_speaker)
            if os.path.exists(dys_file):
                pairs.append({'normal': normal_file, 'dys': dys_file})

        """Make multiprocess data"""
        multiprocess_data = []
        for pair in pairs:
            multiprocess_data.append([pair, dtw_dir, normal_speaker, dys_speaker])

        # dtw_multiprocess(multiprocess_data[0])
        with concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            for _ in tqdm(executor.map(dtw_multiprocess, multiprocess_data)):
                """"""

    def train(self, normal_speaker, dys_speaker, pair_dir):
        solver = Solver(pair_dir, normal_speaker, dys_speaker)
        solver.train_dcgan(pair_dir, normal_speaker, dys_speaker)

    def eval(self, normal_speaker, dys_speaker, pair_dir):
        solver = Solver(pair_dir, normal_speaker, dys_speaker)
        solver.eval_transformer()

def main():
    pairs = get_conversion_pairs()
    """Create directory for time-aligned data (there's a lot)"""
    if not os.path.isdir(config.directories.dtw):
        os.mkdir(config.directories.dtw)
    for pair in pairs:
        """Create pair directory"""
        normal_speaker = pair['source']
        dys_speaker = pair['target']
        pair_dir = os.path.join(exp_root, normal_speaker + '_to_' + dys_speaker)
        if not os.path.isdir(pair_dir):
            os.mkdir(pair_dir)
        print("Working on " + pair_dir)
        process = Process()
        if ALIGN:
            alignment_dir = os.path.join(config.directories.dtw, normal_speaker + '_to_' + dys_speaker)
            if not os.path.isdir(alignment_dir):
                os.mkdir(alignment_dir)
            process.get_dtw(dtw_dir=alignment_dir, normal_speaker=normal_speaker, dys_speaker=dys_speaker)
        if TRAIN:
            process.train(normal_speaker, dys_speaker, pair_dir)
        if EVAL:
            process.eval(normal_speaker, dys_speaker, pair_dir)

# def main_dcgan():
#     for speaker_ in ["M08", "M05", "M04"]:
#         for feature_type_ in ['bap', 'mcep']:
#             normal_speaker = 'CM08'  # same for all three experiments
#             dys_speaker = speaker_
#             FEATURE_TYPE = feature_type_  # mcep or bap
#             hparams = hparams_.get_hparams()
#             EXPERIMENT = 'trial0_' + normal_speaker + '_to_' + dys_speaker + '_' + FEATURE_TYPE
#             device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#             torch_type = torch.float32
#             NUM_EPOCHS = 6000
#             TOTAL_ITERS = 10003  # twice as long as good experiment, should have plenty good results
#             # TOTAL_ITERS = 3  # TEST RUN settings
#
#             """Print hparams"""
#             for param in hparams:
#                 print(param + '=' + str(hparams[param]))
#             """create experiment folder"""
#             if not os.path.isdir('exp/'):
#                 os.mkdir('exp/')
#             if not os.path.isdir('exp/all_models/'):
#                 os.mkdir('exp/all_models/')
#             exp_path = os.path.join('exp/', 'all_models/', EXPERIMENT)
#             if not os.path.isdir(exp_path):
#                 os.mkdir(exp_path)
#             """save hparams"""
#             joblib.dump(hparams, os.path.join(exp_path, 'hparams.pkl'))
#
#
#             partition = joblib.load('partition.pkl')
#
#             """Get paired utterance filepaths"""
#             pairs = pair_features(normal_speaker, dys_speaker, partition)
#
#             """Make dataloader"""
#             train_data = Dataset({'pairs': pairs['train'], 'feat_type': FEATURE_TYPE, "mode": 'train'})
#             train_gen = data.DataLoader(train_data, batch_size=hparams["batch_size"],
#                                         shuffle=True, collate_fn=collate, drop_last=True)
#
#             """create models"""
#             if FEATURE_TYPE == 'mcep':
#                 nfeatures = hparams["mcep_dim"]
#             elif FEATURE_TYPE == 'bap':
#                 nfeatures = hparams["bap_dim"]
#             G = model.Generator()
#             D = model.Discriminator(nfreqs=nfeatures, adapt_size=hparams["adapt_size"])
#             """Put the models on the GPU"""
#             G = (G.cuda()).to(torch_type)
#             D = (D.cuda()).to(torch_type)
#
#             """Initialize BCELoss function"""
#             criterion = nn.BCELoss()
#
#             """Establish convention for real and fake labels during training"""
#             real_label = 1
#             fake_label = 0
#
#             """Setup Adam optimizers for both G and D"""
#             optimizerD = optim.Adam(D.parameters(), lr=hparams["d_lr"], betas=(hparams["beta1"], 0.999))
#             optimizerG = optim.Adam(G.parameters(), lr=hparams["g_lr"], betas=(hparams["beta1"], 0.999))
#
#             total_iters = 0
#             log_path = os.path.join(exp_path, 'logs/')
#             if not os.path.isdir(log_path):
#                 os.mkdir(log_path)
#             from logger import Logger
#             logger_ = Logger(log_path)
#             model_save_dir = os.path.join(exp_path, 'models/')
#             if not os.path.isdir(model_save_dir):
#                 os.mkdir(model_save_dir)
#             image_dir = os.path.join(exp_path, 'images/')
#             if not os.path.isdir(image_dir):
#                 os.mkdir(image_dir)
#             for epoch in range(NUM_EPOCHS):
#                 if total_iters < TOTAL_ITERS:
#                     for batch in train_gen:
#                         """Depending on the length of the sequence, the adapt_size from hparams may be too large and cause error,
#                         but we don't want to crash training. We also want this parameter to be fairly big to retain fine detail
#                         information."""
#                         if total_iters < TOTAL_ITERS:
#                             try:
#                                 normal = (batch["normal"]).to(torch_type)
#                                 dys = (batch["dys"]).to(torch_type)
#                                 """Do a training pass"""
#                                 ############################
#                                 # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
#                                 #     Only update every d_schedule times
#                                 ###########################
#                                 if total_iters % hparams["d_schedule"] == 0:
#                                     """Train with all-real batch"""
#                                     D.zero_grad()
#                                     label = torch.full((hparams["batch_size"],), real_label, device=device)
#                                     """Forward pass real batch through D"""
#                                     output = D(dys).view(-1)
#                                     """Calculate loss on all-real batch"""
#                                     errD_real = criterion(output, label)
#                                     """Calculate gradients for D in backward pass"""
#                                     errD_real.backward()
#                                     D_x = output.mean().item()
#
#                                     """Train with all-fake batch"""
#                                     """Generate fake spectral features batch with G"""
#                                     fake = G(normal)
#                                     label.fill_(fake_label)
#                                     """Classify all fake batch with D"""
#                                     output = D(fake.detach()).view(-1)  # detach so the gradients through G aren't kept (they aren't used to update D)
#                                     """Calculate D's loss on the all-fake batch"""
#                                     errD_fake = criterion(output, label)
#                                     """Calculate the gradients for this batch"""
#                                     errD_fake.backward()
#                                     D_G_z1 = output.mean().item()
#                                     """Add the gradients from the all-real and all-fake batches"""
#                                     errD = errD_real + errD_fake
#                                     """Update D"""
#                                     optimizerD.step()
#                                     # print("Updated D...")
#
#                                 ############################
#                                 # (2) Update G network: maximize log(D(G(z)))
#                                 ###########################
#                                 if total_iters % hparams["d_schedule"] != 0:
#                                     fake = G(normal)  # if we don't calculate from D update step we must do it here
#                                 G.zero_grad()
#                                 label.fill_(real_label)  # fake labels are real for generator cost
#                                 """Since we just updated D, perform another forward pass of all-fake batch through D"""
#                                 output = D(fake).view(-1)
#                                 """Calculate G's loss based on this output"""
#                                 errG = criterion(output, label)
#                                 """Calculate gradients for G"""
#                                 errG.backward()
#                                 D_G_z2 = output.mean().item()
#                                 """Update G"""
#                                 optimizerG.step()
#                                 # print("Updated G...")
#                             except:
#                                 print("Training example too short...")
#
#                             """Output training losses"""
#                             if total_iters % 50 == 0:
#                                 print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
#                                       % (epoch, NUM_EPOCHS, total_iters, len(train_gen),
#                                          errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
#
#                             """Log the losses to tensorboard"""
#                             if total_iters % 10 == 0:
#                                 logger_.scalar_summary('D-loss', errD.item(), total_iters)
#                                 logger_.scalar_summary('G-loss', errG.item(), total_iters)
#
#                             """Save output mceps and dys mceps"""
#                             if total_iters % 200 == 0:
#                                 fake_np = np.squeeze(fake[0].detach().cpu().numpy())  # choose first element in batch
#                                 dys_np = np.squeeze(dys[0].detach().cpu().numpy())  # choose first element in batch
#                                 plt.subplot(211)
#                                 plt.imshow(fake_np[:, 2:].T)
#                                 plt.subplot(212)
#                                 plt.imshow(dys_np[:, 2:].T)
#                                 plt.savefig(os.path.join(image_dir, str(total_iters) + '.png'))
#                                 plt.close()
#
#
#                             if total_iters % 1000 == 0:
#                                 G_path = os.path.join(model_save_dir, '{}-G.ckpt'.format(epoch))
#                                 torch.save({'model': G.state_dict(),
#                                             'optimizer': optimizerG.state_dict()}, G_path)
#                                 D_path = os.path.join(model_save_dir, '{}-D.ckpt'.format(epoch))
#                                 torch.save({'model': D.state_dict(),
#                                             'optimizer': optimizerD.state_dict()}, D_path)
#                                 print('Saved model checkpoints into {}...'.format(model_save_dir))
#
#                             total_iters += 1
#             stop = None

if __name__ == "__main__":
    main()
