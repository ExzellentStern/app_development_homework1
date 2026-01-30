# dcgan_cifar10_from_slide.py
# CIFAR-10 (3ch, 32x32) 対応版 DCGAN

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image  # ← 追加
import numpy as np

# ===== デバイス設定 =====
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===== 出力ディレクトリ（Windowsのデスクトップ直下） =====
home = os.path.expanduser("~")
out_dir = os.path.join(home, "Desktop", "app_development_data")
os.makedirs(out_dir, exist_ok=True)

# ===== ログ出力設定 =====
# ===== ログ出力設定（ターミナル＋ファイル両方に出力） =====
log_path = os.path.join(out_dir, "train_log.txt")
logfile = open(log_path, "a", encoding="utf-8")

# 元のstdout/stderrを保持
console_out = sys.__stdout__
console_err = sys.__stderr__

class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
            except Exception:
                pass
        self.flush()
    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass

# ターミナル＋ファイル両方へ
sys.stdout = Tee(console_out, logfile)
sys.stderr = Tee(console_err, logfile)

print(f"[LOG START] 出力ログ: {log_path}\n", flush=True)

# ===== CIFAR-10 データ =====
batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))   # [-1,1] にスケーリング
])
train_set = datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

# ===== 生成器: z(100) → 3×32×32 =====
class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),  # 1→4
            nn.BatchNorm2d(512), nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),  # 4→8
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),  # 8→16
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False),    # 16→32
            nn.Tanh()
        )
    def forward(self, z):
        return self.net(z)

# ===== 識別器: 3×32×32 → 1 =====
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 128, 4, 2, 1, bias=False),    # 32→16
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),  # 16→8
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),  # 8→4
            nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),    # 4→1
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

G = generator().to(device)
D = discriminator().to(device)

# ===== 損失関数と最適化 =====
criterion = nn.BCELoss()
lr = 2e-4
G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5,0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5,0.999))

# ===== 学習 =====
epochs = 50
fixed_z = torch.randn(64, 100, 1, 1, device=device)  # 進捗比較用の固定ノイズ

for epoch in range(epochs):
    for i, (imgs, _) in enumerate(train_loader):
        bsz = imgs.size(0)

        # ---- Train D ----
        real = imgs.to(device)
        z = torch.randn(bsz, 100, 1, 1, device=device)
        fake = G(z).detach()                     # Gの勾配を切る
        real_out, fake_out = D(real), D(fake)

        D_loss = criterion(torch.cat([real_out, fake_out], 0),
                           torch.cat([torch.ones_like(real_out),
                                      torch.zeros_like(fake_out)], 0))
        D_optimizer.zero_grad()
        D_loss.backward()
        D_optimizer.step()

        # ---- Train G ----
        z = torch.randn(bsz, 100, 1, 1, device=device)
        fake = G(z)
        output = D(fake)
        G_loss = criterion(output, torch.ones_like(output))
        G_optimizer.zero_grad()
        G_loss.backward()
        G_optimizer.step()

        if i % 100 == 0:
            print(f"Epoch {epoch:02d}|D_loss: {D_loss.item():.3f} | G_loss: {G_loss.item():.3f}")

    # ===== エポックごとにサンプル画像を保存 =====
    with torch.no_grad():
        fake_imgs = G(fixed_z).cpu()
    save_path = os.path.join(out_dir, f"epoch_{epoch:03d}.png")
    save_image(fake_imgs, save_path, nrow=8, normalize=True, value_range=(-1, 1))
    print(f"[Saved] {save_path}")

print("\n[LOG END] 学習が完了しました。")
logfile.close()
