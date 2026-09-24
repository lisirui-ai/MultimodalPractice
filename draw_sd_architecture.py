"""
Stable Diffusion VAE & U-Net 架构图（最终修订版）

核心修正：
1. U-Net 采用对称 U 型布局：编码器自上而下，解码器自下而上
   → 每条 skip connection 均完全水平，无交叉
2. skip connections 基于 HuggingFace diffusers 真实实现核对
   （含 conv_in 输出作为 stack[0]，Down-Downsample 跨层到下一 Up 组）
3. VAE 两列对齐，block 高度和字体均适当放大
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

os.makedirs("images", exist_ok=True)
plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ──────────────── 颜色 ────────────────────────────────────────
C = {
    "io":    "#1B3A6B",  # 深蓝   输入/输出
    "conv":  "#1B5E20",  # 深绿   卷积
    "res":   "#7B1A1A",  # 深红   ResNet
    "attn":  "#4A148C",  # 深紫   自注意力
    "cross": "#880E4F",  # 玫红   CrossAttn+Transformer
    "dn":    "#E65100",  # 橙     下采样 Downsample
    "up":    "#006064",  # 深青   上采样 Upsample
    "mid":   "#263238",  # 蓝灰   MidBlock
    "norm":  "#4E342E",  # 棕     GroupNorm+Act
    "time":  "#BF360C",  # 暗橙   时间步嵌入
    "txt":   "#0D47A1",  # 靛蓝   文本条件
    "quant": "#004D40",  # 深绿青 量化卷积
    "rep":   "#4E342E",  # 棕     重参数化
    "sk":    "#D84315",  # 跳跃连接主色（实线，同层）
    "skx":   "#F57F17",  # 跳跃连接次色（虚线，跨层Downsample）
    "bg":    "#F5F5F2",
}


# ════════════════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════════════════

def blk(ax, x, y, w, h, lines, colors=None, base=C["io"],
        fs=9.5, tc="white", radius=0.22, lw=1.4):
    """圆角矩形块，lines=[行文字]，可按行指定颜色(colors)或统一base色。"""
    if colors is None:
        p = FancyBboxPatch((x, y), w, h,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            facecolor=base, edgecolor="white", linewidth=lw, zorder=3)
        ax.add_patch(p)
        n = len(lines)
        for i, ln in enumerate(lines):
            fsi = fs if i == 0 else fs - 1.5
            fw  = "bold" if i == 0 else "normal"
            alp = 1.0   if i == 0 else 0.88
            ax.text(x + w/2, y + h * (1 - (i + 0.55) / n),
                    ln, ha="center", va="center",
                    fontsize=fsi, color=tc, fontweight=fw,
                    alpha=alp, zorder=4)
    else:
        n = len(lines); sh = h / n
        for i, (ln, col) in enumerate(zip(lines, colors)):
            sy = y + (n - 1 - i) * sh
            p = FancyBboxPatch((x, sy), w, sh,
                boxstyle=f"round,pad=0,rounding_size={min(radius,0.15)}",
                facecolor=col, edgecolor="white", linewidth=0.8, zorder=3)
            ax.add_patch(p)
            ax.text(x + w/2, sy + sh/2, ln, ha="center", va="center",
                    fontsize=fs - (0 if i == 0 else 1.2), color="white",
                    fontweight="bold" if i == 0 else "normal",
                    alpha=0.95, zorder=4)


def av(ax, x, y1, y2, col="#555", lw=1.4):
    """垂直箭头 y1→y2。"""
    ax.annotate("", xy=(x, y2), xytext=(x, y1),
        arrowprops=dict(arrowstyle="-|>", color=col, lw=lw,
            mutation_scale=11), zorder=5)


def ah(ax, x1, x2, y, col=C["sk"], lw=1.7, lbl="", dashed=False):
    """水平箭头 x1→x2（skip connection）。"""
    ls = "--" if dashed else "-"
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
        arrowprops=dict(arrowstyle="-|>", color=col, lw=lw,
            linestyle=ls, mutation_scale=10), zorder=5)
    if lbl:
        ax.text((x1+x2)/2, y + 0.09, lbl, ha="center", va="bottom",
                fontsize=7, color=col, zorder=6,
                bbox=dict(fc="white", ec="none", alpha=0.8, pad=0.03))


def ac(ax, x1, y1, x2, y2, col=C["sk"], lw=1.7, rad=0.3, lbl=""):
    """曲线箭头（用于非水平的连线）。"""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=col, lw=lw,
            mutation_scale=11,
            connectionstyle=f"arc3,rad={rad}"), zorder=5)
    if lbl:
        mx, my = (x1*0.45+x2*0.55), (y1*0.45+y2*0.55)
        ax.text(mx, my, lbl, ha="center", va="center", fontsize=7,
                color=col, zorder=6,
                bbox=dict(fc="white", ec=col, alpha=0.9, pad=0.12,
                          boxstyle="round,pad=0.2"))


def grp(ax, x, y, w, h, lbl="", fc="#E8E8E0", ec="#AAAAAA", fs=8.5):
    """分组框（虚线背景）。"""
    p = FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0,rounding_size=0.3",
        facecolor=fc, edgecolor=ec, linewidth=0.9,
        linestyle="--", alpha=0.25, zorder=1)
    ax.add_patch(p)
    if lbl:
        ax.text(x+0.15, y+h-0.1, lbl, ha="left", va="top",
                fontsize=fs, color="#333", fontweight="bold", zorder=2)


def lgd(ax, items, x, y, title="图例"):
    ax.text(x, y, title, ha="left", va="top",
            fontsize=10, fontweight="bold", color="#222")
    for i, (lbl, col) in enumerate(items):
        iy = y - 0.65 - i*0.60
        p = FancyBboxPatch((x, iy-0.21), 0.55, 0.42,
            boxstyle="round,pad=0,rounding_size=0.08",
            facecolor=col, edgecolor="white", linewidth=0.8, zorder=3)
        ax.add_patch(p)
        ax.text(x+0.72, iy, lbl, ha="left", va="center",
                fontsize=8.2, color="#222")


def make_ax(fw, fh):
    fig, ax = plt.subplots(figsize=(fw, fh))
    ax.set_xlim(0, fw); ax.set_ylim(0, fh)
    ax.axis("off")
    ax.set_facecolor(C["bg"]); fig.patch.set_facecolor(C["bg"])
    return fig, ax


# ════════════════════════════════════════════════════════
# VAE 架构图
# ════════════════════════════════════════════════════════

def draw_vae():
    FW, FH = 28, 38
    fig, ax = make_ax(FW, FH)

    BW = 11.0; BH = 1.0; GAP = 0.42
    EL = 0.8; EX = EL + BW/2
    DL = FW - BW - 0.8; DX = DL + BW/2
    step = BH + GAP

    # 标题
    ax.text(FW/2, FH-0.6, "VAE（变分自编码器）架构图",
            ha="center", fontsize=20, fontweight="bold", color="#1A1A2E")
    ax.text(FW/2, FH-1.4,
            "stabilityai/stable-diffusion-2-1-base  ·  AutoencoderKL",
            ha="center", fontsize=10.5, color="#555")
    ax.text(EX, FH-2.2, "① Encoder（编码器）",
            ha="center", fontsize=12.5, fontweight="bold", color=C["io"])
    ax.text(DX, FH-2.2, "② Decoder（解码器）",
            ha="center", fontsize=12.5, fontweight="bold", color=C["up"])

    # ── 编码器步骤 ──────────────────────────────────
    enc = [
        # (主标签, shape文字, 颜色)
        ("输入图像  Input",
         "[B, 3, 512, 512]", C["io"]),
        ("conv_in   Conv2d(3→128, 3×3, pad=1)",
         "[B, 128, 512, 512]", C["conv"]),
        ("DownBlock①  ResNet×2  Conv(128→128)",
         "[B, 128, 512, 512]", C["res"]),
        ("DownBlock①  Downsample  Conv(128→128, stride=2)",
         "[B, 128, 256, 256]", C["dn"]),
        ("DownBlock②  ResNet①  Conv(128→256)  +conv_shortcut",
         "[B, 256, 256, 256]", C["res"]),
        ("DownBlock②  ResNet②  Conv(256→256)",
         "[B, 256, 256, 256]", C["res"]),
        ("DownBlock②  Downsample  Conv(256→256, stride=2)",
         "[B, 256, 128, 128]", C["dn"]),
        ("DownBlock③  ResNet①  Conv(256→512)  +conv_shortcut",
         "[B, 512, 128, 128]", C["res"]),
        ("DownBlock③  ResNet②  Conv(512→512)",
         "[B, 512, 128, 128]", C["res"]),
        ("DownBlock③  Downsample  Conv(512→512, stride=2)",
         "[B, 512, 64, 64]", C["dn"]),
        ("DownBlock④  ResNet×2  Conv(512→512)  [无下采样]",
         "[B, 512, 64, 64]", C["res"]),
        ("MidBlock  ResNet-M1  Conv(512→512)",
         "[B, 512, 64, 64]", C["mid"]),
        ("MidBlock  SelfAttn  Q/K/V: Linear(512→512)",
         "[B, 512, 64, 64]", C["attn"]),
        ("MidBlock  ResNet-M2  Conv(512→512)",
         "[B, 512, 64, 64]", C["mid"]),
        ("GroupNorm(32,512)  +  SiLU",
         "[B, 512, 64, 64]", C["norm"]),
        ("conv_out  Conv2d(512→8, 3×3, pad=1)",
         "[B, 8, 64, 64]", C["conv"]),
        ("quant_conv  Conv2d(8→8, 1×1)  分离 μ / log σ²",
         "[B, 8, 64, 64]", C["quant"]),
    ]

    ey = []; cur = FH - 3.1 - BH
    for i, (t1, t2, col) in enumerate(enc):
        blk(ax, EL, cur, BW, BH, [t1, t2], base=col, fs=9.2)
        ey.append(cur)
        if i > 0: av(ax, EX, ey[-2]-0.02, ey[-1]+BH+0.02)
        cur -= step

    grp(ax, EL-.3, ey[2]-.2, BW+.6, (ey[2]-ey[3])+BH+.4,
        "DownBlock①  128ch  512→256", "#EEE8DD")
    grp(ax, EL-.3, ey[4]-.2, BW+.6, (ey[4]-ey[6])+BH+.4,
        "DownBlock②  128→256ch  256→128", "#EEE8DD")
    grp(ax, EL-.3, ey[7]-.2, BW+.6, (ey[7]-ey[9])+BH+.4,
        "DownBlock③  256→512ch  128→64", "#EEE8DD")
    grp(ax, EL-.3, ey[10]-.2, BW+.6, BH+.4,
        "DownBlock④  512ch  64×64  无下采样", "#EEE8DD")
    grp(ax, EL-.3, ey[11]-.2, BW+.6, (ey[11]-ey[13])+BH+.4,
        "MidBlock  512ch  64×64", "#E0D8F0")

    # 重参数化
    qy = ey[16]  # quant_conv bottom-y
    ax.text(EX, qy-0.52,
            "↓ 分割：前4通道 = μ  /  后4通道 = log σ²",
            ha="center", fontsize=9.5, color="#5D4037", fontweight="bold",
            bbox=dict(fc="#FFF8E1", ec="#FF8F00", lw=1.1,
                      boxstyle="round,pad=0.3"))
    rpy = qy - 1.4
    sw = BW/2 - 0.3
    blk(ax, EL+0.1, rpy, sw, BH,
        ["μ  均值", "[B, 4, 64, 64]"], base=C["rep"], fs=9.5)
    blk(ax, EL+sw+0.5, rpy, sw, BH,
        ["log σ²  对数方差", "[B, 4, 64, 64]"], base=C["rep"], fs=9.5)
    ac(ax, EX-.4, qy-0.95, EL+0.1+sw/2, rpy+BH,
       col="#8B6914", lw=1.4, rad=-0.2)
    ac(ax, EX+.4, qy-0.95, EL+sw+0.5+sw/2, rpy+BH,
       col="#8B6914", lw=1.4, rad=0.2)

    zy = rpy - 1.35
    ax.text(EX, rpy-0.58,
            "重参数化：z = μ + σ·ε，  ε ~ N(0, I)",
            ha="center", fontsize=10, color="#5D4037", fontweight="bold",
            bbox=dict(fc="#FFF8E1", ec="#FF8F00", lw=1.1,
                      boxstyle="round,pad=0.32"))
    blk(ax, EL+2.0, zy, BW-4.0, BH,
        ["latent  z  (采样)", "[B, 4, 64, 64]"], base=C["io"], fs=10)
    av(ax, EX, rpy-0.92, zy+BH, col="#5D4037")

    # ── 解码器步骤 ──────────────────────────────────
    dec = [
        ("latent z 输入", "[B, 4, 64, 64]", C["io"]),
        ("post_quant_conv  Conv2d(4→4, 1×1)",
         "[B, 4, 64, 64]", C["quant"]),
        ("conv_in  Conv2d(4→512, 3×3, pad=1)",
         "[B, 512, 64, 64]", C["conv"]),
        ("MidBlock  ResNet-M1  Conv(512→512)",
         "[B, 512, 64, 64]", C["mid"]),
        ("MidBlock  SelfAttn  Q/K/V: Linear(512→512)",
         "[B, 512, 64, 64]", C["attn"]),
        ("MidBlock  ResNet-M2  Conv(512→512)",
         "[B, 512, 64, 64]", C["mid"]),
        ("UpBlock①  ResNet×3  Conv(512→512)",
         "[B, 512, 64, 64]", C["res"]),
        ("UpBlock①  Upsample  插值×2 + Conv(512→512)",
         "[B, 512, 128, 128]", C["up"]),
        ("UpBlock②  ResNet×3  Conv(512→512)",
         "[B, 512, 128, 128]", C["res"]),
        ("UpBlock②  Upsample  插值×2 + Conv(512→512)",
         "[B, 512, 256, 256]", C["up"]),
        ("UpBlock③  ResNet①  Conv(512→256)+conv_shortcut",
         "[B, 256, 256, 256]", C["res"]),
        ("UpBlock③  ResNet②③  Conv(256→256)",
         "[B, 256, 256, 256]", C["res"]),
        ("UpBlock③  Upsample  插值×2 + Conv(256→256)",
         "[B, 256, 512, 512]", C["up"]),
        ("UpBlock④  ResNet①  Conv(256→128)+conv_shortcut",
         "[B, 128, 512, 512]", C["res"]),
        ("UpBlock④  ResNet②③  Conv(128→128)",
         "[B, 128, 512, 512]", C["res"]),
        ("GroupNorm(32,128)  +  SiLU",
         "[B, 128, 512, 512]", C["norm"]),
        ("conv_out  Conv2d(128→3, 3×3, pad=1)",
         "[B, 3, 512, 512]", C["conv"]),
        ("输出图像  Output", "[B, 3, 512, 512]", C["io"]),
    ]

    dy = []; cur = FH - 3.1 - BH
    for i, (t1, t2, col) in enumerate(dec):
        blk(ax, DL, cur, BW, BH, [t1, t2], base=col, fs=9.2)
        dy.append(cur)
        if i > 0: av(ax, DX, dy[-2]-0.02, dy[-1]+BH+0.02)
        cur -= step

    grp(ax, DL-.3, dy[3]-.2, BW+.6, (dy[3]-dy[5])+BH+.4,
        "MidBlock  512ch  64×64", "#E0D8F0")
    grp(ax, DL-.3, dy[6]-.2, BW+.6, (dy[6]-dy[7])+BH+.4,
        "UpBlock①  512ch  64→128", "#D8EDD8")
    grp(ax, DL-.3, dy[8]-.2, BW+.6, (dy[8]-dy[9])+BH+.4,
        "UpBlock②  512ch  128→256", "#D8EDD8")
    grp(ax, DL-.3, dy[10]-.2, BW+.6, (dy[10]-dy[12])+BH+.4,
        "UpBlock③  512→256ch  256→512", "#D8EDD8")
    grp(ax, DL-.3, dy[13]-.2, BW+.6, (dy[13]-dy[14])+BH+.4,
        "UpBlock④  256→128ch  无上采样", "#D8EDD8")

    # latent z → decoder 连线
    ax.annotate("", xy=(DL-.08, dy[0]+BH/2),
                xytext=(EL+BW+.08, zy+BH/2),
                arrowprops=dict(arrowstyle="-|>", color=C["io"], lw=2.0,
                    mutation_scale=13, connectionstyle="arc3,rad=0"), zorder=5)
    ax.text((EL+BW+DL)/2, (zy+dy[0])/2+0.4,
            "latent  z\n[B, 4, 64, 64]",
            ha="center", fontsize=10.5, color=C["io"], fontweight="bold",
            bbox=dict(fc="#E3EAF8", ec=C["io"], lw=1.4,
                      boxstyle="round,pad=0.42", alpha=0.92))

    lgd(ax, [
        ("输入 / 输出", C["io"]),
        ("卷积层 Conv", C["conv"]),
        ("残差块 ResNet", C["res"]),
        ("自注意力 SelfAttn", C["attn"]),
        ("下采样 Downsample", C["dn"]),
        ("上采样 Upsample", C["up"]),
        ("瓶颈层 MidBlock", C["mid"]),
        ("GroupNorm + 激活", C["norm"]),
        ("量化卷积 QuantConv", C["quant"]),
        ("重参数化 z=μ+σε", C["rep"]),
    ], x=12.5, y=dy[-1]-0.4)

    plt.tight_layout(pad=0.3)
    out = "images/vae_architecture.png"
    plt.savefig(out, dpi=160, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"VAE 架构图已保存：{out}")
    return out


# ════════════════════════════════════════════════════════
# U-Net 架构图  —— 对称 U 型布局
# ════════════════════════════════════════════════════════
#
# 编码器 stack（LIFO 消费）已验证：
#   stack[ 0] conv_in 输出         [B, 320, 64,  64]
#   stack[ 1] Down0-ResNet0+Attn0  [B, 320, 64,  64]
#   stack[ 2] Down0-ResNet1+Attn1  [B, 320, 64,  64]
#   stack[ 3] Down0-Downsample     [B, 320, 32,  32]   ← 跨层 → Up2-R2
#   stack[ 4] Down1-ResNet0+Attn0  [B, 640, 32,  32]
#   stack[ 5] Down1-ResNet1+Attn1  [B, 640, 32,  32]
#   stack[ 6] Down1-Downsample     [B, 640, 16,  16]   ← 跨层 → Up1-R2
#   stack[ 7] Down2-ResNet0+Attn0  [B,1280, 16,  16]
#   stack[ 8] Down2-ResNet1+Attn1  [B,1280, 16,  16]
#   stack[ 9] Down2-Downsample     [B,1280,  8,   8]   ← 跨层 → Up0-R2
#   stack[10] Down3-ResNet0        [B,1280,  8,   8]
#   stack[11] Down3-ResNet1        [B,1280,  8,   8]
#
# 对应关系（全部水平对齐，无交叉）：
#   行号 | 编码器 (从高到低存入 stack)  | 解码器 (从低到高消费 stack)
#    15  | conv_in          [B,320,64,64] | Up3-R2  ← stack[ 0]
#    14  | Down0-ResNet0+A  [B,320,64,64] | Up3-R1  ← stack[ 1]
#    13  | Down0-ResNet1+A  [B,320,64,64] | Up3-R0  ← stack[ 2]
#   US2  | [连接线]                        | Up2-Upsample
#    11  | Down0-Downsample [B,320,32,32] | Up2-R2  ← stack[ 3] ★跨层
#    10  | Down1-ResNet0+A  [B,640,32,32] | Up2-R1  ← stack[ 4]
#     9  | Down1-ResNet1+A  [B,640,32,32] | Up2-R0  ← stack[ 5]
#   US1  | [连接线]                        | Up1-Upsample
#     7  | Down1-Downsample [B,640,16,16] | Up1-R2  ← stack[ 6] ★跨层
#     6  | Down2-ResNet0+A  [B,1280,16,16]| Up1-R1  ← stack[ 7]
#     5  | Down2-ResNet1+A  [B,1280,16,16]| Up1-R0  ← stack[ 8]
#   US0  | [连接线]                        | Up0-Upsample
#     3  | Down2-Downsample [B,1280,8,8]  | Up0-R2  ← stack[ 9] ★跨层
#     2  | Down3-ResNet0    [B,1280,8,8]  | Up0-R1  ← stack[10]
#     1  | Down3-ResNet1    [B,1280,8,8]  | Up0-R0  ← stack[11]
#         MidBlock（底部）
#
# ════════════════════════════════════════════════════════

def draw_unet():
    FW, FH = 34, 42
    fig, ax = make_ax(FW, FH)

    SBH = 0.90   # 子块高度
    GAP = 0.42   # 子块间距
    step = SBH + GAP
    BW   = 12.0  # 块宽
    EL   = 0.5;  EX = EL + BW/2
    DL   = FW - BW - 0.5; DX = DL + BW/2

    # ── 从顶部向下计算布局 ─────────────────────────────────
    # 标题: FH-0.6, FH-1.4
    # 时间步/文本嵌入条: emb_y
    # 列标题: emb_y - 0.8
    # 编码器 Input: top_area_y
    # conv_in (row15 = pos14): top_area_y - step
    # row14..row0: 依次向下
    # MidBlock: 最底部

    emb_y = FH - 3.2     # 时间步嵌入 bottom-y（从顶往下）
    COL_HDR_Y = emb_y - 1.0
    INP_Y  = COL_HDR_Y - 0.9   # Input block bottom-y
    TOP_Y  = INP_Y - step       # conv_in (row15, pos14) bottom-y

    # pos → bottom-y（从 pos=14 最高到 pos=0 最低）
    def ry(pos):
        return TOP_Y - (14 - pos) * step

    MID_H = 3.0 * step
    MID_BOT_Y = ry(0) - GAP*2 - MID_H  # MidBlock 底部 y

    # ── 标题区 ─────────────────────────────────────────────
    ax.text(FW/2, FH-0.6,
            "U-Net 去噪网络架构图（UNet2DConditionModel）",
            ha="center", fontsize=18, fontweight="bold", color="#1A1A2E")
    ax.text(FW/2, FH-1.4,
            "stabilityai/stable-diffusion-2-1-base  ·  文本条件去噪",
            ha="center", fontsize=10.5, color="#555")

    # 时间步嵌入 & 文本嵌入 条框
    blk(ax, 0.3, emb_y, FW*0.43, SBH,
        ["时间步嵌入   t → sincos → Linear(320→1280→1280)",
         "[B,320→1280]  注入每个 ResNet 的 time_emb_proj"],
        base=C["time"], fs=9.0)
    blk(ax, FW*0.43+0.6, emb_y, FW*0.54, SBH,
        ["文本嵌入（CLIPTextModel 输出）  作为 CrossAttn 的 K / V",
         "[B, 77, 1024]  送入所有 CrossAttn 层"],
        base=C["txt"], fs=9.0)

    ax.text(EX, COL_HDR_Y,
            "⬇  编码器 Encoder（下采样路径）",
            ha="center", va="top", fontsize=11.5, fontweight="bold",
            color=C["io"])
    ax.text(DX, COL_HDR_Y,
            "⬆  解码器 Decoder（上采样路径）",
            ha="center", va="top", fontsize=11.5, fontweight="bold",
            color=C["up"])

    # ── 顶部：Input & conv_in (编码器) ──────────────────────
    blk(ax, EL, INP_Y, BW, SBH,
        ["latent 输入  x_t  (带噪 latent)",
         "[B, 4, 64, 64]"],
        base=C["io"], fs=9.5)

    # conv_in（pos=14，最高行）
    blk(ax, EL, TOP_Y, BW, SBH,
        ["conv_in  Conv2d(4→320, 3×3, pad=1)  →  保存 stack[0]",
         "[B, 320, 64, 64]"],
        base=C["conv"], fs=9.2)
    av(ax, EX, INP_Y-0.02, TOP_Y+SBH+0.02)   # Input → conv_in（向下）

    # ── 顶部：解码器输出块（GN+SiLU, conv_out, Output）────────
    GN_Y   = TOP_Y                # GroupNorm+SiLU 与 conv_in 同高（顶行）
    COUT_Y = INP_Y                # conv_out 与 Input 同高
    OUT_Y  = INP_Y + step         # Output 在 Input 上方

    blk(ax, DL, GN_Y, BW, SBH,
        ["GroupNorm(32, 320)  +  SiLU",
         "[B, 320, 64, 64]"],
        base=C["norm"], fs=9.2)
    blk(ax, DL, COUT_Y, BW, SBH,
        ["conv_out  Conv2d(320→4, 3×3, pad=1)",
         "[B, 4, 64, 64]"],
        base=C["conv"], fs=9.2)
    blk(ax, DL, OUT_Y, BW, SBH,
        ["噪声预测  ε_θ(x_t, t, c_text)  Output",
         "[B, 4, 64, 64]"],
        base=C["io"], fs=9.5)
    # 解码器顶部箭头（向上）
    av(ax, DX, GN_Y+SBH+0.02, COUT_Y-0.02)   # GN → conv_out（向上）
    av(ax, DX, COUT_Y+SBH+0.02, OUT_Y-0.02)  # conv_out → Output（向上）

    # ── 行定义 ─────────────────────────────────────────────
    # 每行：(pos, enc_spec_or_None, dec_spec, is_skip, skip_label, skip_color, is_cross_level)
    # enc_spec: (label1, label2, color) or None（Upsample 行无编码器块）
    # dec_spec: (label1, label2, color)
    # is_cross_level: True 表示 Downsample 跨层 skip（用虚线）

    ROWS = [
        # pos=14 → row 15 (64×64, conv_in level)
        (14,
         None,  # 编码器在此行已经单独画了 conv_in（row15）
         ("Up3-ResNet2+CrossAttn  Conv(640→320)  ← stack[0]  conv_in",
          "[B, 320, 64, 64]   cat=[320+320]=640→320  ch=64×64",
          C["cross"]),
         True, "stack[0]\n[B,320,64,64]", C["skx"], True),

        # pos=13 → row 14
        (13,
         ("Down0-ResNet0+CrossAttn  [保存 stack[1]]",
          "[B, 320, 64, 64]   ch=320, 64×64",
          C["cross"]),
         ("Up3-ResNet1+CrossAttn  Conv(640→320)  ← stack[1]",
          "[B, 320, 64, 64]   cat=[320+320]=640→320",
          C["cross"]),
         True, "stack[1]\n[B,320,64,64]", C["sk"], False),

        # pos=12 → row 13
        (12,
         ("Down0-ResNet1+CrossAttn  [保存 stack[2]]",
          "[B, 320, 64, 64]   ch=320, 64×64",
          C["cross"]),
         ("Up3-ResNet0+CrossAttn  Conv(960→320)  ← stack[2]",
          "[B, 320, 64, 64]   cat=[640+320]=960→320",
          C["cross"]),
         True, "stack[2]\n[B,320,64,64]", C["sk"], False),

        # pos=11 → US2: Up2-Upsample（编码器无块，仅连线）
        (11,
         None,
         ("Up2-Upsample  插值上采样×2  +  Conv(640→640)",
          "[B, 640, 64, 64]  32→64",
          C["up"]),
         False, "", C["sk"], False),

        # pos=10 → row 11 (32×32, Down0-Downsample)
        (10,
         ("Down0-Downsample  Conv(320→320, stride=2)  [保存 stack[3]]",
          "[B, 320, 32, 32]   下采样 64→32",
          C["dn"]),
         ("Up2-ResNet2+CrossAttn  Conv(960→640)  ← stack[3]  ★跨层",
          "[B, 640, 32, 32]   cat=[640+320]=960→640",
          C["cross"]),
         True, "stack[3]\n[B,320,32,32]\n★跨层", C["skx"], True),

        # pos=9 → row 10
        (9,
         ("Down1-ResNet0+CrossAttn  [保存 stack[4]]",
          "[B, 640, 32, 32]   ch=640, 32×32",
          C["cross"]),
         ("Up2-ResNet1+CrossAttn  Conv(1280→640)  ← stack[4]",
          "[B, 640, 32, 32]   cat=[640+640]=1280→640",
          C["cross"]),
         True, "stack[4]\n[B,640,32,32]", C["sk"], False),

        # pos=8 → row 9
        (8,
         ("Down1-ResNet1+CrossAttn  [保存 stack[5]]",
          "[B, 640, 32, 32]   ch=640, 32×32",
          C["cross"]),
         ("Up2-ResNet0+CrossAttn  Conv(1920→640)  ← stack[5]",
          "[B, 640, 32, 32]   cat=[1280+640]=1920→640",
          C["cross"]),
         True, "stack[5]\n[B,640,32,32]", C["sk"], False),

        # pos=7 → US1: Up1-Upsample
        (7,
         None,
         ("Up1-Upsample  插值上采样×2  +  Conv(1280→1280)",
          "[B, 1280, 32, 32]  16→32",
          C["up"]),
         False, "", C["sk"], False),

        # pos=6 → row 7 (16×16, Down1-Downsample)
        (6,
         ("Down1-Downsample  Conv(640→640, stride=2)  [保存 stack[6]]",
          "[B, 640, 16, 16]   下采样 32→16",
          C["dn"]),
         ("Up1-ResNet2+CrossAttn  Conv(1920→1280)  ← stack[6]  ★跨层",
          "[B, 1280, 16, 16]  cat=[1280+640]=1920→1280",
          C["cross"]),
         True, "stack[6]\n[B,640,16,16]\n★跨层", C["skx"], True),

        # pos=5 → row 6
        (5,
         ("Down2-ResNet0+CrossAttn  [保存 stack[7]]",
          "[B, 1280, 16, 16]  ch=1280, 16×16",
          C["cross"]),
         ("Up1-ResNet1+CrossAttn  Conv(2560→1280)  ← stack[7]",
          "[B, 1280, 16, 16]  cat=[1280+1280]=2560→1280",
          C["cross"]),
         True, "stack[7]\n[B,1280,16,16]", C["sk"], False),

        # pos=4 → row 5
        (4,
         ("Down2-ResNet1+CrossAttn  [保存 stack[8]]",
          "[B, 1280, 16, 16]  ch=1280, 16×16",
          C["cross"]),
         ("Up1-ResNet0+CrossAttn  Conv(2560→1280)  ← stack[8]",
          "[B, 1280, 16, 16]  cat=[1280+1280]=2560→1280",
          C["cross"]),
         True, "stack[8]\n[B,1280,16,16]", C["sk"], False),

        # pos=3 → US0: Up0-Upsample
        (3,
         None,
         ("Up0-Upsample  插值上采样×2  +  Conv(1280→1280)",
          "[B, 1280, 16, 16]  8→16",
          C["up"]),
         False, "", C["sk"], False),

        # pos=2 → row 3 (8×8, Down2-Downsample)
        (2,
         ("Down2-Downsample  Conv(1280→1280, stride=2)  [保存 stack[9]]",
          "[B, 1280, 8, 8]    下采样 16→8",
          C["dn"]),
         ("Up0-ResNet2  Conv(2560→1280)  ← stack[9]  ★跨层  (无 CrossAttn)",
          "[B, 1280, 8, 8]    cat=[1280+1280]=2560→1280",
          C["res"]),
         True, "stack[9]\n[B,1280,8,8]\n★跨层", C["skx"], True),

        # pos=1 → row 2
        (1,
         ("Down3-ResNet0  Conv(1280→1280)  [保存 stack[10]]  无 CrossAttn",
          "[B, 1280, 8, 8]    ch=1280, 8×8",
          C["res"]),
         ("Up0-ResNet1  Conv(2560→1280)  ← stack[10]  (无 CrossAttn)",
          "[B, 1280, 8, 8]    cat=[1280+1280]=2560→1280",
          C["res"]),
         True, "stack[10]\n[B,1280,8,8]", C["sk"], False),

        # pos=0 → row 1
        (0,
         ("Down3-ResNet1  Conv(1280→1280)  [保存 stack[11]]  无 CrossAttn",
          "[B, 1280, 8, 8]    ch=1280, 8×8  (编码器最深)",
          C["res"]),
         ("Up0-ResNet0  Conv(2560→1280)  ← stack[11]  (无 CrossAttn)",
          "[B, 1280, 8, 8]    cat=[1280+1280]=2560→1280",
          C["res"]),
         True, "stack[11]\n[B,1280,8,8]", C["sk"], False),
    ]

    # ── 绘制各行 ─────────────────────────────────────────────
    enc_ys = {}  # pos → bottom_y（编码器）
    dec_ys = {}  # pos → bottom_y（解码器）

    for row in ROWS:
        pos, enc_spec, dec_spec, do_skip, skip_lbl, skip_col, _ = row
        by = ry(pos)  # block bottom-y

        # 解码器块（始终有）
        blk(ax, DL, by, BW, SBH,
            [dec_spec[0], dec_spec[1]], base=dec_spec[2], fs=8.8)
        dec_ys[pos] = by

        # 编码器块（Upsample 行无）
        if enc_spec is not None:
            blk(ax, EL, by, BW, SBH,
                [enc_spec[0], enc_spec[1]], base=enc_spec[2], fs=8.8)
            enc_ys[pos] = by

        # Skip connection 箭头（水平）
        if do_skip and skip_lbl:
            ey = by + SBH/2
            dy_mid = by + SBH/2
            ah(ax, EL+BW+0.06, DL-0.06, ey,
               col=skip_col, lw=1.7, lbl=skip_lbl,
               dashed=(skip_col == C["skx"]))

    # ── 垂直连线（编码器，自上而下）─────────────────────────
    # conv_in (TOP_Y) → row15(pos14) → row14(pos13) → ... → row1(pos0) → MidBlock
    prev_e = TOP_Y  # conv_in bottom-y
    for pos in [14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]:
        if pos in enc_ys:
            av(ax, EX, prev_e-0.02, enc_ys[pos]+SBH+0.02, col="#444")
            prev_e = enc_ys[pos]
        else:
            # Upsample 行：编码器画一条连接线（无块）
            next_enc = [enc_ys[p] for p in [pos-1] if p in enc_ys]
            if next_enc:
                av(ax, EX, prev_e-0.02, next_enc[0]+SBH+0.02, col="#444")
                prev_e = next_enc[0]

    # ── 垂直连线（解码器，自下而上）────────────────────────────
    # MidBlock → row1(pos0) → ... → row15(pos14) → GN+SiLU(TOP_Y)
    prev_d = MID_BOT_Y + MID_H  # MidBlock top-y
    for pos in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]:
        if pos in dec_ys:
            av(ax, DX, prev_d+0.02, dec_ys[pos]-0.02, col="#444")
            prev_d = dec_ys[pos]+SBH

    # GN+SiLU (TOP_Y) 往上的解码器连线
    av(ax, DX, prev_d+0.02, TOP_Y-0.02, col="#444")
    av(ax, DX, TOP_Y+SBH+0.02, INP_Y-0.02, col="#444")

    # ── MidBlock ───────────────────────────────────────────
    MX = (EX + DX)/2  # MidBlock 中心 x

    mid_specs = [
        ("MidBlock-ResNet-M1  Conv(1280→1280)  +  time_emb_proj",
         "[B, 1280, 8, 8]", C["mid"]),
        ("MidBlock-Transformer:  SelfAttn(1280)  +  CrossAttn(Q:1280, K/V:1024→1280)  +  GEGLU FFN",
         "[B, 1280, 8, 8]   在 8×8 最小分辨率融合文本条件", C["cross"]),
        ("MidBlock-ResNet-M2  Conv(1280→1280)  +  time_emb_proj",
         "[B, 1280, 8, 8]", C["mid"]),
    ]
    MW2 = FW - 2.0; ML2 = 1.0
    mid_ys_list = []
    cy = MID_BOT_Y + (MID_H - len(mid_specs)*step)/2
    for i, (t1, t2, col) in enumerate(mid_specs):
        blk(ax, ML2, cy, MW2, SBH, [t1, t2], base=col, fs=9.0)
        mid_ys_list.append(cy)
        if i > 0: av(ax, FW/2, mid_ys_list[-2]-0.02,
                     mid_ys_list[-1]+SBH+0.02)
        cy -= step

    grp(ax, ML2-0.3, MID_BOT_Y-0.2,
        MW2+0.6, MID_H+0.4,
        "UNetMidBlock2DCrossAttn  [ch=1280, 8×8 最小分辨率]",
        fc="#ECEFF1", ec="#455A64", fs=9.5)

    # 编码器末端 → MidBlock 入口
    ac(ax, EX, enc_ys.get(0, ry(0))+(-0.05),
       ML2 + MW2*0.3, mid_ys_list[0]+SBH,
       col=C["mid"], lw=2.0, rad=0.35,
       lbl="编码器\n→ MidBlock")

    # MidBlock 出口 → 解码器起点
    ac(ax, ML2+MW2*0.7, mid_ys_list[-1]+SBH,
       DL, dec_ys.get(0, ry(0))+SBH,
       col=C["mid"], lw=2.0, rad=-0.35,
       lbl="MidBlock\n→ 解码器")

    # ── 分组背景 ──────────────────────────────────────────
    # 编码器分组
    def enc_grp(p_top, p_bot, lbl, fc, ec):
        y_top = enc_ys.get(p_top, ry(p_top)) + SBH
        y_bot = enc_ys.get(p_bot, ry(p_bot))
        grp(ax, EL-0.3, y_bot-0.2, BW+0.6,
            y_top-y_bot+0.4, lbl, fc=fc, ec=ec, fs=8)

    if 14 in enc_ys:
        enc_grp(12, 14, "CrossAttnDownBlock①  [ch=320, 64×64→32×32]",
                "#EEF2FF", "#3949AB")
    if 10 in enc_ys:
        enc_grp(9, 10, "CrossAttnDownBlock②  [ch=320→640, 32→16]",
                "#E8F5E9", "#388E3C")
    if 6 in enc_ys:
        enc_grp(5, 6, "CrossAttnDownBlock③  [ch=640→1280, 16→8]",
                "#F3E5F5", "#7B1FA2")
    if 0 in enc_ys and 1 in enc_ys:
        enc_grp(1, 0, "DownBlock④  [ch=1280, 8×8  无CrossAttn无Downsample]",
                "#FBE9E7", "#D84315")

    # 解码器分组
    def dec_grp(p_top, p_bot, lbl, fc, ec):
        y_top = dec_ys.get(p_top, ry(p_top)) + SBH
        y_bot = dec_ys.get(p_bot, ry(p_bot))
        grp(ax, DL-0.3, y_bot-0.2, BW+0.6,
            y_top-y_bot+0.4, lbl, fc=fc, ec=ec, fs=8)

    dec_grp(3, 0, "UpBlock①  [ch=1280, 8→16  无CrossAttn]",
            "#FBE9E7", "#D84315")
    dec_grp(7, 4, "CrossAttnUpBlock②  [ch=1280, 16→32]",
            "#F3E5F5", "#7B1FA2")
    dec_grp(11, 8, "CrossAttnUpBlock③  [ch=1280→640, 32→64]",
            "#E8F5E9", "#388E3C")
    dec_grp(14, 12, "CrossAttnUpBlock④  [ch=640→320, 64×64  无Upsample]",
            "#EEF2FF", "#3949AB")

    # ── 图例 ──────────────────────────────────────────────
    lgd(ax, [
        ("输入 / 输出", C["io"]),
        ("卷积层 Conv", C["conv"]),
        ("残差块 ResNet (无CrossAttn)", C["res"]),
        ("自注意力 SelfAttn", C["attn"]),
        ("CrossAttn + Transformer", C["cross"]),
        ("下采样 Downsample", C["dn"]),
        ("上采样 Upsample", C["up"]),
        ("MidBlock", C["mid"]),
        ("GroupNorm + 激活", C["norm"]),
        ("时间步嵌入 TimeEmb", C["time"]),
        ("文本条件 TextEmb", C["txt"]),
        ("Skip (同层)", C["sk"]),
        ("Skip (★跨层 Downsample)", C["skx"]),
    ], x=EL+BW*0.1, y=MID_BOT_Y-0.5, title="▌ 图例")

    # CrossAttn 说明
    ca_x = EL + BW + 0.8
    ca_y = MID_BOT_Y - 0.5
    ax.text(ca_x, ca_y,
            "CrossAttn 机制\n"
            "  Q: Linear(latent_ch → latent_ch)  ← 图像 latent\n"
            "  K: Linear(1024 → latent_ch)         ← 文本嵌入\n"
            "  V: Linear(1024 → latent_ch)         ← 文本嵌入\n"
            "  Attn = softmax(QKᵀ/√d)·V\n"
            "  FFN: GEGLU，升维4×再降维",
            ha="left", va="top", fontsize=8.8, color="#1A1A2E",
            bbox=dict(fc="#E8F0FF", ec=C["cross"], lw=1.2,
                      boxstyle="round,pad=0.45", alpha=0.92))

    plt.tight_layout(pad=0.3)
    out = "images/unet_architecture.png"
    plt.savefig(out, dpi=160, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"U-Net 架构图已保存：{out}")
    return out


# ════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("绘制 VAE 架构图...")
    draw_vae()
    print("绘制 U-Net 架构图...")
    draw_unet()
    print("完成。")
