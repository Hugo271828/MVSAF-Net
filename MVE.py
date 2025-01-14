
class PatchEmbed2D(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        deform_groups = 1

        # 第一条路径: 标准卷积
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        # 第二条路径: 扩张卷积
        self.proj2 = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, dilation=2)

        # 第三条路径: 可变形卷积
        self.offset_conv = nn.Conv2d(in_chans, deform_groups * 2 * patch_size[0] * patch_size[1],
                                     kernel_size=patch_size, stride=patch_size, padding=1)
        self.deform_conv = DeformConv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, padding=1, deform_groups=deform_groups)

        # 注意力机制卷积
        self.attention_conv = nn.Conv2d(embed_dim * 3, 3, 1)

        # 为六条路径生成门控信号的卷积层
        self.gate_conv1_forward = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.gate_conv2_forward = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.gate_conv3_forward = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.gate_conv1_reverse = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.gate_conv2_reverse = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.gate_conv3_reverse = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        # 第一条路径: 标准卷积
        x1 = self.proj(x)
        x1 = F.interpolate(x1, size=(128, 128), mode='bilinear', align_corners=False)

        # 第二条路径: 扩张卷积
        x2 = self.proj2(x)
        x2 = F.interpolate(x2, size=(128, 128), mode='bilinear', align_corners=False)

        # 第三条路径: 可变形卷积
        offset = self.offset_conv(x)
        offset = offset.contiguous()
        x=x.contiguous()
        x3 = self.deform_conv(x, offset)
        x3 = F.interpolate(x3, size=(128, 128), mode='bilinear', align_corners=False)

        # 拼接三条路径的输出
        concat = torch.cat([x1, x2, x3], dim=1)
        attn_scores = self.attention_conv(concat)
        attn_weights = F.softmax(attn_scores, dim=1)

        # 正向注意力
        attn1, attn2, attn3 = torch.chunk(attn_weights, chunks=3, dim=1)

        # 反向注意力
        reverse_attn1 = 1 - attn1
        reverse_attn2 = 1 - attn2
        reverse_attn3 = 1 - attn3

        # 为六条路径生成门控信号
        gate1_forward = torch.sigmoid(self.gate_conv1_forward(x1))  # 第一条路径正向的门控信号
        gate2_forward = torch.sigmoid(self.gate_conv2_forward(x2))  # 第二条路径正向的门控信号
        gate3_forward = torch.sigmoid(self.gate_conv3_forward(x3))  # 第三条路径正向的门控信号

        gate1_reverse = torch.sigmoid(self.gate_conv1_reverse(x1))  # 第一条路径反向的门控信号
        gate2_reverse = torch.sigmoid(self.gate_conv2_reverse(x2))  # 第二条路径反向的门控信号
        gate3_reverse = torch.sigmoid(self.gate_conv3_reverse(x3))  # 第三条路径反向的门控信号

        # 三条正向路径的加权输出
        x1_forward = gate1_forward * (attn1 * x1)
        x2_forward = gate2_forward * (attn2 * x2)
        x3_forward = gate3_forward * (attn3 * x3)

        # 三条反向路径的加权输出
        x1_reverse = gate1_reverse * (reverse_attn1 * x1)
        x2_reverse = gate2_reverse * (reverse_attn2 * x2)
        x3_reverse = gate3_reverse * (reverse_attn3 * x3)

        # 将正向和反向路径的输出相加
        out = (x1_forward + x2_forward + x3_forward) + (x1_reverse + x2_reverse + x3_reverse)

        # 调整输出形状并进行归一化
        out = out.permute(0, 2, 3, 1)  # (B, H, W, C)
        if self.norm is not None:
            out = self.norm(out)
        return out
