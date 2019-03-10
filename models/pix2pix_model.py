import torch
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import numpy as np

caffe_root='/home/cunjian/Documents/research/other_projects/GenerativeFaceCompletion/'
import sys
sys.path.insert(0,caffe_root+'python')
import caffe
caffe.set_device(0)
caffe.set_mode_gpu()

# define model
model_def=caffe_root+'matlab/FaceCompletion_testing/model/Model_parsing.prototxt'
model_weights=caffe_root+'matlab/FaceCompletion_testing/model/Model_parsing.caffemodel'
net = caffe.Net(model_def, model_weights, caffe.TEST)


class Pix2PixModel(BaseModel):
    def name(self):
        return 'Pix2PixModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G_GAN', 'G_L1', 'G_VGGFace','G_VGG', 'D_real', 'D_fake','G_S']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G']
        else:  # during test time, only load Gs
            self.model_names = ['G']
        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionVGG = networks.VGGLoss(self.gpu_ids)
            self.criterionVGGFace = networks.VGGFaceLoss(self.gpu_ids)

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.which_epoch)

        self.print_networks(opt.verbose)

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0], async=True)
            input_B = input_B.cuda(self.gpu_ids[0], async=True)
        self.input_A = input_A
        self.input_B = input_B
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.fake_B = self.netG(self.real_A)
        self.real_B = Variable(self.input_B)

    # no backprop gradients
    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.fake_B = self.netG(self.real_A)
        self.real_B = Variable(self.input_B, volatile=True)

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1))
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A

        # define perceptual loss
        self.loss_G_VGG = self.criterionVGG(self.fake_B, self.real_B) * 10

        # define vggface loss
        self.loss_G_VGGFace = self.criterionVGGFace(self.fake_B, self.real_B) * 20

        # define semantic loss
        fake_B_sample=self.fake_B.data.cpu().numpy()
        fake_B_sample.shape = (3,256,256)
        fake_B_sample = np.transpose(fake_B_sample, [1,2,0]) 
        fake_B_sample=fake_B_sample[::2,::2,:]
        #print(fake_B_sample)
        input_ = fake_B_sample.transpose(2, 1, 0)
        input_ = input_[np.newaxis, ...]
        net.blobs['data'].reshape(*input_.shape)
        net.blobs['data'].data[...] = input_
        output = net.forward()
        scores = output['conv_decode0'][0]
        scores = scores.transpose(0, 2, 1) # swap x and y axis
        #scores = scores[4:10,:,:]
        scores = scores.argmax(axis=0) # convert to class
        scores[scores==1] = 0
        scores[scores==2] = 0
        scores[scores!=0] = 1
        #print(scores.shape)
        #scores = Variable(torch.from_numpy(scores)).cuda().
        scores = Variable(torch.from_numpy(scores)).cuda().float()

        # real sample
        real_B_sample=self.real_B.data.cpu().numpy()
        real_B_sample.shape = (3,256,256)
        real_B_sample = np.transpose(real_B_sample, [1,2,0]) 
        real_B_sample=real_B_sample[::2,::2,:]
        #print(fake_B_sample)
        input_real = real_B_sample.transpose(2, 1, 0)
        input_real = input_real[np.newaxis, ...]
        net.blobs['data'].reshape(*input_real.shape)
        net.blobs['data'].data[...] = input_real
        output_real = net.forward()
        scores_real = output_real['conv_decode0'][0]
        scores_real = scores_real.transpose(0, 2, 1) # swap x and y axis
        #scores_real = scores_real[4:10,:,:]
        scores_real = scores_real.argmax(axis=0)
        scores_real[scores_real==1] = 0
        scores_real[scores_real==2] = 0
        scores_real[scores_real!=0] = 1
        scores_real = Variable(torch.from_numpy(scores_real)).cuda().float()

        self.loss_G_S = self.criterionL1(scores,scores_real)*20
        #print(self.loss_G_S)


        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_VGG + self.loss_G_VGGFace + self.loss_G_S


        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
