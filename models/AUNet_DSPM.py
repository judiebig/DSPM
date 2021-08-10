"""
AUNet: a classical Attention-UNet designed in DARCN (main network)
"""


import torch
import torch.nn as nn
from torch.autograd import Variable
import os
from models.DSPM import DSPM
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class AUNet_DSPM(nn.Module):
    def __init__(self):
        super(AUNet_DSPM, self).__init__()
        # Main Encoder Part
        self.en = Encoder()
        self.de = Decoder()
        self.glu_list = nn.ModuleList([GLU(dilation=2 ** 0, in_channel=256) for i in range(6)])

    def forward(self, x):
        ori = x
        x = x.unsqueeze(dim=1)
        x, en_list = self.en(x)
        batch_size, _, seq_len, _ = x.shape
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(batch_size, -1, seq_len)
        x_skip = Variable(torch.zeros(x.shape), requires_grad=True).to(x.device)
        for i in range(6):
            x = self.glu_list[i](x)
            x_skip = x_skip + x
        x = x_skip
        x = x.view(batch_size, 64, 4, seq_len)
        x = x.permute(0, 1, 3, 2).contiguous()
        x = self.de(x, en_list)
        del x_skip, en_list
        return x.squeeze()


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.pad1 = nn.ConstantPad2d((0, 0, 1, 0), value=0.)
        self.pad2 = nn.ConstantPad2d((1, 1, 1, 0), value=0.)
        self.pad3 = nn.ConstantPad2d((2, 2, 1, 0), value=0.)
        self.fen1 = nn.Sequential(
            self.pad3,
            DSPM(1, 16, kernel_size=(2, 5), num_experts=3, stride=(1, 1)))
        self.ben1 = nn.Sequential(
            nn.BatchNorm2d(16),
            nn.ELU())
        self.fen2 = nn.Sequential(
            self.pad1,
            DSPM(16, 16, kernel_size=(2, 5), num_experts=3, stride=(1, 2)))
        self.ben2 = nn.Sequential(
            nn.BatchNorm2d(16),
            nn.ELU())
        self.fen3 = nn.Sequential(
            self.pad2,
            DSPM(16, 32, kernel_size=(2, 5), num_experts=3, stride=(1, 2)))
        self.ben3 = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ELU())
        self.fen4 = nn.Sequential(
            self.pad2,
            DSPM(32, 32, kernel_size=(2, 5), num_experts=3, stride=(1, 2)))
        self.ben4 = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ELU())
        self.fen5 = nn.Sequential(
            self.pad2,
            DSPM(32, 64, kernel_size=(2, 5), num_experts=3, stride=(1, 2)))
        self.ben5 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ELU())
        self.en6 = nn.Sequential(
            self.pad2,
            DSPM(64, 64, kernel_size=(2, 5), num_experts=3, stride=(1, 2)),
            nn.BatchNorm2d(64),
            nn.ELU())

    def forward(self, x):
        x_list = []
        x = self.fen1(x)
        x = self.ben1(x)
        x = self.fen2(x)
        x = self.ben2(x)
        x_list.append(x)
        x = self.fen3(x)
        x = self.ben3(x)
        x_list.append(x)
        x = self.fen4(x)
        x = self.ben4(x)
        x_list.append(x)
        x = self.fen5(x)
        x = self.ben5(x)
        x_list.append(x)
        x = self.en6(x)
        x_list.append(x)
        return x, x_list

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.up_f = up_Chomp_F(1)
        self.down_f = down_Chomp_F(1)
        self.chomp_t = Chomp_T(1)
        self.att1 = Attention_Block(64, 64, 64)
        self.att2 = Attention_Block(64, 64, 64)
        self.att3 = Attention_Block(32, 32, 32)
        self.att4 = Attention_Block(32, 32, 32)
        self.att5 = Attention_Block(16, 16, 16)
        self.de1 = nn.Sequential(
            nn.ConvTranspose2d(64*2, out_channels=64, kernel_size=(2, 5), stride=(1, 2)),
            self.up_f,
            self.down_f,
            self.chomp_t,
            nn.BatchNorm2d(64),
            nn.ELU())
        self.de2 = nn.Sequential(
            nn.ConvTranspose2d(64*2, 32, kernel_size=(2, 5), stride=(1, 2)),
            self.up_f,
            self.down_f,
            self.chomp_t,
            nn.BatchNorm2d(32),
            nn.ELU())
        self.de3 = nn.Sequential(
            nn.ConvTranspose2d(32*2, 32, kernel_size=(2, 5), stride=(1, 2)),
            self.up_f,
            self.down_f,
            self.chomp_t,
            nn.BatchNorm2d(32),
            nn.ELU())
        self.de4 = nn.Sequential(
            nn.ConvTranspose2d(32*2, 16, kernel_size=(2, 5), stride=(1, 2)),
            self.up_f,
            self.down_f,
            self.chomp_t,
            nn.BatchNorm2d(16),
            nn.ELU())
        self.de5 = nn.Sequential(
            nn.ConvTranspose2d(16*2, 16, kernel_size=(2, 5), stride=(1, 2)),
            self.chomp_t,
            nn.BatchNorm2d(16),
            nn.ELU())
        self.de6 = nn.Sequential(
            nn.ConvTranspose2d(16, 1, kernel_size=(1,1), stride=(1, 1)),
            nn.Softplus())

    def forward(self, x, en_list):
        en_list[-1] = self.att1(x, en_list[-1])
        x = self.de1(torch.cat((x, en_list[-1]), dim=1))
        en_list[-2] = self.att2(x, en_list[-2])
        x = self.de2(torch.cat((x, en_list[-2]), dim=1))
        en_list[-3] = self.att3(x, en_list[-3])
        x = self.de3(torch.cat((x, en_list[-3]), dim=1))
        en_list[-4] = self.att4(x, en_list[-4])
        x = self.de4(torch.cat((x, en_list[-4]), dim=1))
        en_list[-5] = self.att5(x, en_list[-5])
        x = self.de5(torch.cat((x, en_list[-5]), dim=1))
        x = self.de6(x)
        return x



class Attention_Block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_Block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x*psi

class up_Chomp_F(nn.Module):
    def __init__(self, chomp_f):
        super(up_Chomp_F, self).__init__()
        self.chomp_f = chomp_f

    def forward(self, x):
        return x[:, :, :, self.chomp_f:]


class down_Chomp_F(nn.Module):
    def __init__(self, chomp_f):
        super(down_Chomp_F, self).__init__()
        self.chomp_f = chomp_f

    def forward(self, x):
        return x[:, :, :, :-self.chomp_f]


class Chomp_T(nn.Module):
    def __init__(self, chomp_t):
        super(Chomp_T, self).__init__()
        self.chomp_t = chomp_t

    def forward(self, x):
        return x[:, :, :-self.chomp_t, :]

class GLU(nn.Module):
    def __init__(self, dilation, in_channel):
        super(GLU, self).__init__()
        self.in_conv = nn.Sequential(
            nn.Conv1d(in_channel, 64, kernel_size=1),
            nn.BatchNorm1d(64))
        self.pad = nn.ConstantPad1d((int(dilation * 5), int(dilation * 5)), value=0.)

        self.left_conv = nn.Sequential(
            nn.ELU(),
            self.pad,
            nn.Conv1d(64, 64, kernel_size=11, dilation=dilation),
            nn.BatchNorm1d(64))
        self.right_conv = nn.Sequential(
            nn.ELU(),
            self.pad,
            nn.Conv1d(64, 64, kernel_size=11, dilation=dilation),
            nn.BatchNorm1d(num_features=64),
            nn.Sigmoid())
        self.out_conv = nn.Sequential(
            nn.Conv1d(64, 256, kernel_size=1),
            nn.BatchNorm1d(256))
        self.out_elu = nn.ELU()

    def forward(self, inpt):
        x = inpt
        x = self.in_conv(x)
        x1 = self.left_conv(x)
        x2 = self.right_conv(x)
        x = x1 * x2
        x = self.out_conv(x)
        x = x + inpt
        x = self.out_elu(x)
        return x