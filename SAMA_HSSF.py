class ContextBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 ratio,
                 pooling_type='att',
                 fusion_types=('channel_mul', )):
        super(ContextBlock, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.pooling_type == 'att':
            kaiming_init(self.conv_mask, mode='fan_in')
            self.conv_mask.inited = True

        if self.channel_add_conv is not None:
            last_zero_init(self.channel_add_conv)
        if self.channel_mul_conv is not None:
            last_zero_init(self.channel_mul_conv)

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        out = x
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out + out * channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out


class ConvBranch(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, 3, padding=1, groups=hidden_features, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, 1, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, 3, padding=1, groups=hidden_features, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, 1, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.SiLU(inplace=True)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, 3, padding=1, groups=hidden_features, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.ReLU(inplace=True)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.ca = MS_CAM(64)
        self.sa = SpatialAttention()
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        res1 = x
        res2 = x
        x = self.conv1(x)
        x = x + self.conv2(x)
        x = self.conv3(x)
        x = x + self.conv4(x)
        x = self.conv5(x)
        x = x + self.conv6(x)
        x = self.conv7(x)
        x_mask = self.sigmoid_spatial(x)
        res1 = res1 * x_mask
        return res2 + res1


###GLSANet: Global-Local Self-Attention Network for Remote Sensing Image Semantic Segmentation
class GLSA(nn.Module):
    def __init__(self, input_dim=512, embed_dim=96, k_s=3):
        super().__init__()

        self.conv1_1 = BasicConv2d(embed_dim * 2, embed_dim, 1)
        self.conv1_1_1 = BasicConv2d(input_dim // 2, embed_dim, 1)
        self.local_11conv = nn.Conv2d(input_dim // 2, embed_dim, 1)
        self.global_11conv = nn.Conv2d(input_dim // 2, embed_dim, 1)
        self.GlobelBlock = ContextBlock(inplanes=embed_dim, ratio=2)
        self.local = ConvBranch(in_features=embed_dim, hidden_features=embed_dim, out_features=embed_dim)

    def forward(self, x):
        b, c, h, w = x.size()
        x_0, x_1 = x.chunk(2, dim=1)

        # local block
        local = self.local(self.local_11conv(x_0))

        # Globel block
        Globel = self.GlobelBlock(self.global_11conv(x_1))

        # concat Globel + local
        x = torch.cat([local, Globel], dim=1)
        x = self.conv1_1(x)

        return x

##MS-CAM: Multi-Scale Class Activation Maps for Weakly-Supervised Segmentation of Geographic Atrophy Lesions in SD-OCT Images
class MS_CAM(nn.Module):
    '''
    单特征进行通道注意力加权,作用类似SE模块
    '''

    def __init__(self, channels=64, r=4):
        super(MS_CAM, self).__init__()
        inter_channels = int(channels // r)

        # 局部注意力
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 全局注意力
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # senet中池化
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        return x * wei




class SDI(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.convs = nn.ModuleList(
            [nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1) for _ in range(4)])

    def forward(self, xs, anchor):
        ans = torch.ones_like(anchor)
        target_size = anchor.shape[-1]

        for i, x in enumerate(xs):#[f1,f2,f3,f4]
            if x.shape[-1] > target_size:
                x = F.adaptive_avg_pool2d(x, (target_size, target_size))
            elif x.shape[-1] < target_size:
                x = F.interpolate(x, size=(target_size, target_size),
                                      mode='bilinear', align_corners=True)

            ans = ans * self.convs[i](x)

        return ans

##from VMUnetV2
class VMUNetV2(nn.Module):
    def __init__(self,
                 input_channels=3,
                 num_classes=1,
                 mid_channel=48,
                 depths=[2, 2, 9, 2],
                 depths_decoder=[2, 9, 2, 2],
                 drop_path_rate=0.2,
                 load_ckpt_path=None,
                 deep_supervision=True
                 ):
        super().__init__()

        self.load_ckpt_path = load_ckpt_path
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision

        # SDI
        self.ca_1 = MS_CAM(2 * mid_channel)
        self.sa_1 = GLSA(2 * mid_channel, 2 * mid_channel)

        self.ca_2 = MS_CAM(4 * mid_channel)
        self.sa_2 = GLSA(4 * mid_channel, 4 * mid_channel)
        # TODO 320 or mid_channel * 8?
        self.ca_3 = MS_CAM(8 * mid_channel)
        self.sa_3 = GLSA(8 * mid_channel, 8 * mid_channel)

        self.ca_4 = MS_CAM(16 * mid_channel)
        self.sa_4 = GLSA(16 * mid_channel, 16 * mid_channel)

        self.Translayer_1 = BasicConv2d(2 * mid_channel, mid_channel, 1)
        self.Translayer_2 = BasicConv2d(4 * mid_channel, mid_channel, 1)
        self.Translayer_3 = BasicConv2d(8 * mid_channel, mid_channel, 1)
        self.Translayer_4 = BasicConv2d(16 * mid_channel, mid_channel, 1)

        self.sdi_1 = SDI(mid_channel)
        self.sdi_2 = SDI(mid_channel)
        self.sdi_3 = SDI(mid_channel)
        self.sdi_4 = SDI(mid_channel)

        self.seg_outs = nn.ModuleList([
            nn.Conv2d(mid_channel, num_classes, 1, 1) for _ in range(4)])

        self.deconv2 = nn.ConvTranspose2d(mid_channel, mid_channel, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv3 = nn.ConvTranspose2d(mid_channel, mid_channel, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv4 = nn.ConvTranspose2d(mid_channel, mid_channel, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv5 = nn.ConvTranspose2d(mid_channel, mid_channel, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv6 = nn.ConvTranspose2d(1, 1, 3, stride=2, padding=1, output_padding=1)

        self.vmunet = VSSM(in_chans=input_channels,
                           num_classes=num_classes,
                           depths=depths,
                           depths_decoder=depths_decoder,
                           drop_path_rate=drop_path_rate,
                           )

    def forward(self, x):
        seg_outs = []
        if x.size()[1] == 1:  # 如果是灰度图，就将1个channel 转为3个channel
            x = x.repeat(1, 3, 1, 1)
        f1, f2, f3, f4, x = self.vmunet(x)  # f1 [2, 64, 64, 96]  f3  [2, 8, 8, 768]  [b h w c]
        # b h w c --> b c h w
        f1 = f1.permute(0, 3, 1, 2)  # f1 [2, 96, 64, 64]
        f2 = f2.permute(0, 3, 1, 2)  # f2 [8,192,64,64]
        f3 = f3.permute(0, 3, 1, 2)  # f3 [8,384,32,32]
        f4 = f4.permute(0, 3, 1, 2)  # f4 [8,768,16,16]

        # use sdi
        f1 = self.ca_1(f1) * f1  # f1 [8, 96, 128, 128]
        f1 = self.sa_1(f1)
        f1 = self.Translayer_1(f1)  # f1 [2, 48, 64, 64]

        f2 = self.ca_2(f2) * f2
        f2 = self.sa_2(f2)
        f2 = self.Translayer_2(f2)  # f2 [2, 48, 32, 32]

        f3 = self.ca_3(f3) * f3
        f3 = self.sa_3(f3)
        f3 = self.Translayer_3(f3)  # f3 [2, 48, 16, 16]

        f4 = self.ca_4(f4) * f4
        f4 = self.sa_4(f4)
        f4 = self.Translayer_4(f4)  # f4 [2, 48, 8, 8]

        # # 可视化每个跳跃链接
        # visualize_skip_connection(f1, 'f1')
        # visualize_skip_connection(f2, 'f2')
        # visualize_skip_connection(f3, 'f3')
        # visualize_skip_connection(f4, 'f4')

        f41 = self.sdi_4([f1, f2, f3, f4], f4)  # [2, 48, 8, 8]
        # visualize_skip_connection(f41, 'f41')
        f31 = self.sdi_3([f1, f2, f3, f4], f3)  # [2, 48, 16, 16]
        # visualize_skip_connection(f31, 'f31')
        f21 = self.sdi_2([f1, f2, f3, f4], f2)  # [2, 48, 32, 32]
        # visualize_skip_connection(f21, 'f21')
        f11 = self.sdi_1([f1, f2, f3, f4], f1)  # [2, 48, 64, 64]
        # visualize_skip_connection(f11, 'f11')

        # 函数seg_outs 输出列表也是 seg_outs 只是名字相同
        seg_outs.append(self.seg_outs[0](f41))  # seg_outs[0] [2, 1, 8, 8]

        y = self.deconv2(f41) + f31
        seg_outs.append(self.seg_outs[1](y))  # seg_outs[1] [2, 1, 16, 16]

        y = self.deconv3(y) + f21
        seg_outs.append(self.seg_outs[2](y))  # seg_outs[2] [2, 1, 32, 32]

        y = self.deconv4(y) + f11
        seg_outs.append(self.seg_outs[3](y))  # seg_outs[3] [2, 1, 64, 64]

        for i, o in enumerate(seg_outs):  # 4 倍上采样
            seg_outs[i] = F.interpolate(o, scale_factor=4, mode='bilinear')

        for i, o in enumerate(seg_outs):  # 4 倍上采样
            seg_outs[i] = F.interpolate(o, scale_factor=4, mode='bilinear')

        if self.deep_supervision:

            temp = seg_outs[::-1]  # 0 [2, 1, 256, 256] 1 [2, 1, 128, 128]
            out_0 = temp[0]
            out_1 = temp[1]
            out_1 = self.deconv6(out_1)
            # visualize_skip_connection(out_1, 'orrin_last')
            # visualize_skip_connection(out_1, 'supervison')
            # visualize_skip_connection(torch.sigmoid(out_0 + out_1), 'final')

            out = torch.sigmoid(out_0 + out_1)
            out = F.interpolate(out, size=(512, 512), mode='bilinear', align_corners=False)
            return out  # [2, 1, 256, 256]
        else:
            if self.num_classes == 1:
                return torch.sigmoid(seg_outs[-1])
            else:
                return seg_outs[-1]