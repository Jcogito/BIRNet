from math import log
import torch.nn.functional as F
from net.SwinTransformer import *


class CBR3X3(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(CBR3X3, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=dilation, dilation=dilation, bias=bias),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class CBR1x1(nn.Module):
    def __init__(self, inplanes, planes, bias=True):
        super(CBR1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1, bias=bias)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return x


class deconv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, strides=2):
        super(deconv2d, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels,
                                        kernel_size=kernel_size,
                                        stride=strides, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        return out


class AP_MP(nn.Module):
    def __init__(self, channel):
        super(AP_MP, self).__init__()
        self.convs_apmp = CBR3X3(channel, channel)
        self.gapLayer = nn.AvgPool2d(kernel_size=2, stride=2)
        self.gmpLayer = nn.MaxPool2d(kernel_size=2, stride=2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.convs_apmp(x)
        apimg = self.gapLayer(x1)
        mpimg = self.gmpLayer(x1)
        byimg = torch.norm(abs(apimg-mpimg), p=2, dim=1, keepdim=True)
        edge = F.interpolate(byimg, scale_factor=2, mode='bilinear', align_corners=False)
        edge_att = self.sigmoid(edge)
        return x * edge_att


class CCM(nn.Module):
    def __init__(self, inchannel, channel):
        super(CCM, self).__init__()
        self.conv1d = nn.Sequential(CBR1x1(inchannel, channel), CBR3X3(channel, channel))
        self.conv1d_e = nn.Sequential(CBR1x1(inchannel, channel), CBR3X3(channel, channel))

    def forward(self, x):
        x_a = self.conv1d(x)
        x_e = self.conv1d_e(x)
        return x_a, x_e


class GGBlock(nn.Module):
    def __init__(self, channel):
        super(GGBlock, self).__init__()
        self.block = nn.Conv2d(channel, 1, 7, 1, 3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x4 = self.sigmoid(self.block(x))
        x3 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x2 = F.interpolate(x4, scale_factor=4, mode='bilinear', align_corners=False)
        x1 = F.interpolate(x4, scale_factor=8, mode='bilinear', align_corners=False)
        return x4, x3, x2, x1


class Res2NetBlock(nn.Module):
    def __init__(self, channel, inchannel):
        super(Res2NetBlock, self).__init__()
        self.conv1_1 = CBR1x1(channel, channel)
        self.conv3_1 = CBR3X3(inchannel, inchannel, 3)
        self.dconv5_1 = CBR3X3(inchannel, inchannel, 3)
        self.dconv7_1 = CBR3X3(inchannel, inchannel, 3)
        self.dconv9_1 = CBR3X3(inchannel, inchannel, 3)
        self.conv1_2 = CBR1x1(channel, channel)
        self.convs = CBR3X3(channel, channel)

    def forward(self, cur):
        x = self.conv1_1(cur)
        xc = torch.chunk(x, 4, dim=1)
        x0 = self.conv3_1(xc[0])
        x1 = self.dconv5_1(xc[1] + x0)
        x2 = self.dconv7_1(xc[2] + x1)
        x3 = self.dconv9_1(xc[3] + x2)
        xx = self.conv1_2(torch.cat((x0, x1, x2, x3), dim=1)) + cur
        return self.convs(xx)


class EEM(nn.Module):
    def __init__(self, channel):
        super(EEM, self).__init__()
        self.res2net = Res2NetBlock(channel, channel // 4)
        self.apmp = AP_MP(channel)

    def forward(self, x):
        e1 = self.res2net(x)
        edge = self.apmp(e1)
        return edge


class OEM(nn.Module):
    def __init__(self, channel):
        super(OEM, self).__init__()
        self.mam = Res2NetBlock(channel, channel // 4)
        t = int(abs((log(channel, 2) + 1) / 2))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.mam(x)
        w = self.avg_pool(x1)
        w = self.conv1d(w.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        w = self.sigmoid(w)
        xx = x1 * w
        return xx



class Ubone(nn.Module):
    def __init__(self, inchannel, channel):  # 拼接融合
        super(Ubone, self).__init__()
        self.deconv = deconv2d(inchannel, channel)
        self.deconv_e = deconv2d(inchannel, channel)
        self.convs = nn.Sequential(CBR3X3(2 * channel, channel), CBR3X3(channel, channel))
        self.convs_e = nn.Sequential(CBR3X3(2 * channel, channel), CBR3X3(channel, channel))

    def forward(self, x, x_lat, e, e_lat):
        x_lat = self.deconv(x_lat)
        xx = torch.cat((x, x_lat), 1)
        x_cur = self.convs(xx)
        e_lat = self.deconv_e(e_lat)
        ee = torch.cat((e, e_lat), 1)
        e_cur = self.convs_e(ee)
        return x_cur, e_cur


# 通道对齐
class BOI(nn.Module):
    def __init__(self, channel):
        super(BOI, self).__init__()
        self.convs1x1_1 = CBR1x1(channel, channel // 2)
        self.convs1x1_2 = CBR1x1(channel, channel // 2)
        self.channel = channel // 2
        self.linear = nn.Linear(channel // 2, channel // 2, bias=False)
        self.block = CBR3X3(channel, channel)

    def forward(self, a, e):
        fa = self.convs1x1_1(a)
        fe = self.convs1x1_2(e)
        feature_size = fa.size()[2:]
        dim = feature_size[0] * feature_size[1]
        fa_flat = fa.view(-1, self.channel, dim)
        fe_flat = fe.view(-1, self.channel, dim)
        fa_transpose = torch.transpose(fa_flat, 1, 2).contiguous()
        A = torch.bmm(fe_flat, fa_transpose)
        A = self.linear(A)

        A1 = F.softmax(A.clone(), dim=2)
        A2 = F.softmax(torch.transpose(A, 1, 2), dim=2)
        fa_att = torch.bmm(A1, fa_flat).contiguous()
        fe_att = torch.bmm(A2, fe_flat).contiguous()
        fa_att = fa_att.view(-1, self.channel, feature_size[0], feature_size[1])
        fa_out = fa_att + fa
        fe_att = fe_att.view(-1, self.channel, feature_size[0], feature_size[1])
        fe_out = fe_att + fe
        out = self.block(torch.cat((fa_out, fe_out), 1))
        return out


class BTC(nn.Module):
    def __init__(self, channel):
        super(BTC, self).__init__()
        self.aff = BOI(channel)
        self.sa1 = nn.Sequential(nn.Conv2d(channel, 1, 7, padding=3, bias=True), nn.Sigmoid())
        self.sa2 = nn.Sequential(nn.Conv2d(channel, 1, 7, padding=3, bias=True), nn.Sigmoid())
        self.cbam = OEM(channel)
        self.apmp = EEM(channel)
        self.block_a = nn.Sequential(CBR3X3(2*channel, channel), CBR3X3(channel, channel))
        self.block_e = nn.Sequential(CBR3X3(2*channel, channel), CBR3X3(channel, channel))

    def forward(self, x, xe, att):
        sali = self.cbam(x)
        edge = self.apmp(xe)
        aff = self.aff(sali, edge)
        sali_cur = torch.cat((sali, aff.mul(self.sa1(sali))), 1)
        sali_out = self.block_a(sali_cur) + x.mul(att)
        edge_cur = torch.cat((edge, aff.mul(self.sa2(edge))), 1)
        edge_out = self.block_e(edge_cur) + xe.mul(att)
        return sali_out, edge_out


class decoder(nn.Module):
    def __init__(self, k):
        super(decoder, self).__init__()
        self.predictor_s = nn.Conv2d(k, 1, 1)
        self.predictor_e = nn.Conv2d(k, 1, 1)

    def forward(self, x, e):
        s = self.predictor_s(x)
        e = self.predictor_e(e)
        return s, e


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.swin = SwinTransformer(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32])

        self.ggm = GGBlock(1024)

        self.ccm4 = CCM(1024, 512)
        self.ccm3 = CCM(512, 256)
        self.ccm2 = CCM(256, 128)
        self.ccm1 = CCM(128, 64)

        self.seim4 = BTC(512)
        self.seim3 = BTC(256)
        self.seim2 = BTC(128)
        self.seim1 = BTC(64)

        self.bone3 = Ubone(512, 256)
        self.bone2 = Ubone(256, 128)
        self.bone1 = Ubone(128, 64)

        self.decoder4 = decoder(512)
        self.decoder3 = decoder(256)
        self.decoder2 = decoder(128)
        self.decoder1 = decoder(64)

    def forward(self, x):
        rgb_list = self.swin(x)
        x1 = rgb_list[0]
        x2 = rgb_list[1]
        x3 = rgb_list[2]
        x4 = rgb_list[3]

        att4, att3, att2, att1 = self.ggm(x4)

        x_4, e_4 = self.ccm4(x4)
        x_3, e_3 = self.ccm3(x3)
        x_2, e_2 = self.ccm2(x2)
        x_1, e_1 = self.ccm1(x1)

        x4a, x4e = self.seim4(x_4, e_4, att4)
        ya_3, ye_3 = self.bone3(x_3, x4a, e_3, x4e)
        x3a, x3e = self.seim3(ya_3, ye_3, att3)
        ya_2, ye_2 = self.bone2(x_2, x3a, e_2, x3e)
        x2a, x2e = self.seim2(ya_2, ye_2, att2)
        ya_1, ye_1 = self.bone1(x_1, x2a, e_1, x2e)
        x1a, x1e = self.seim1(ya_1, ye_1, att1)

        o4, e4 = self.decoder4(x4a, x4e)
        o3, e3 = self.decoder3(x3a, x3e)
        o2, e2 = self.decoder2(x2a, x2e)
        o1, e1 = self.decoder1(x1a, x1e)

        e4 = F.interpolate(e4, scale_factor=32, mode='bilinear', align_corners=False)
        e3 = F.interpolate(e3, scale_factor=16, mode='bilinear', align_corners=False)
        e2 = F.interpolate(e2, scale_factor=8, mode='bilinear', align_corners=False)
        e1 = F.interpolate(e1, scale_factor=4, mode='bilinear', align_corners=False)

        o4 = F.interpolate(o4, scale_factor=32, mode='bilinear', align_corners=False)
        o3 = F.interpolate(o3, scale_factor=16, mode='bilinear', align_corners=False)
        o2 = F.interpolate(o2, scale_factor=8, mode='bilinear', align_corners=False)
        o1 = F.interpolate(o1, scale_factor=4, mode='bilinear', align_corners=False)
        return o4, o3, o2, o1, e4, e3, e2, e1

    def load_pre(self, pre_model):
        self.swin.load_state_dict(torch.load(pre_model)['model'], strict=False)
        print(f"RGB SwinTransformer loading pre_model ${pre_model}")


if __name__ == '__main__':
    model_test = Net()
    input = torch.ones(8, 3, 384, 384)
    output = model_test(input)
    print(output)

