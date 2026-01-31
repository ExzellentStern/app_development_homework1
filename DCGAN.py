

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image  
import numpy as np
import matplotlib.pyplot as plt


# デバイスとしてもしcudaが疲れるならcudaを用い,そうでないならcpuを用いる
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 出力するディレクトリの設定.(DCGAN自体とは関係ない)(私がレポートのために結果を保存したいだけ)
#今ログインしているユーザーのホームディレクトリ(~)の絶対パスを取得する.
#OSというOS関連の機能をまとめた標準ライブラリのパス専用モジュールの中の,実際にホームディレクトリの絶対パスを専用するexpanduser関数ということ.
home = os.path.expanduser("~")
#homeに代入した絶対パスに""の中のフォルダを結合し,output_dirとして出力先を作っている.
out_dir = os.path.join(home, "Desktop", "app_development_data")
#指定したパスのディレクトリを作成する。ただし、すでに存在していてもエラーにしない
os.makedirs(out_dir, exist_ok=True)
#だから絶対パスを手に入れてそこに作りたいフォルダのパスを結合してそのパス宛にフォルダを作ってるということ.

#ログ出力設定
#ここもout_dir（出力先）にtrain_log.txtというパスを結合しているだけ.(ファイル自体はまだ作ってない)
log_path = os.path.join(out_dir, "train_log.txt")
#↑で作ったlog_pathをopenして"append"する.それをlogfileの値として格納する.utf-8は日本語ログが文字化けしないために用いる.
logfile = open(log_path, "a", encoding="utf-8")

# 元のstdout/stderrを保持
#python起動時からの出力先(sys.__stdout__)とエラーの出力先(sys.__stderr__)
#上ではlogfileに出力しているが下ではコンソールに対して出力を行っている.
console_out = sys.__stdout__
console_err = sys.__stderr__
#Teeは出力先を束ねるオブジェクト
class Tee:
    #初期メソッド
    def __init__(self, *streams):
        self.streams = streams
    #write メソッドは、print() によって生成された文字列を受け取り、登録されたすべての出力先へ同一内容を書き込む役割を担う
    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
            except Exception:
                pass
        self.flush()
        #flush メソッドは、各出力先のバッファを即座に書き出すことで、学習途中でプログラムが終了した場合でもログが確実に保存されるようにしている。
    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass

#print() の出力先を、「ターミナル＋ログファイル」の両方に変更している
sys.stdout = Tee(console_out, logfile)
sys.stderr = Tee(console_err, logfile)
#ログの開始メッセージ
print(f"[LOG START] 出力ログ: {log_path}\n", flush=True)
#多分私がデータの保守用に作ったんだろうけど,両方に出力する必要あるのだろうか.
#ファイル出力だけで済むような気がするのだが.


#↑はファイルの設定
##==============================================
#↓データの取得,整形の設定

# CIFAR-10 データの設定.SVHNでもバッチサイズは同じで大丈夫だと思う
#バッチサイズを64に設定.
batch_size = 64
#composeは上から順番に命令を実行する仕組み.ToTensorでPytorchが扱える画像に変換する.
#それを正規化(normalize)する.結果的に入力範囲は[0~1]が[-1~1]に変換される.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))   # [-1,1] にスケーリング
])
#CIFAR-10の画像のデータセットを持ってきて,transformメソッドを用いてそれをpytorchで扱える形に変換しているということ.
#CIFAR-10は訓練用と検証用で分かれているのでtrain=Trueというのは訓練用データを持ってくるという処理.
#dawnloadはもしデータがなかったらダウンロードするという意味.ローカルに存在しない場合のみダウンロードされる
#train_set = datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
#CIFAR-10の訓練データをミニバッチ(batch_size)でシャッフルしながら取り出す(Load)してそれをtrain_loaderに格納する.
#train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

#最終課題ではSVHNを用いる.(画像のサイズが同じでコードにあまり手を加えなくても良いため.)
train_set = datasets.SVHN(root="data", split="train", download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)



#==================================
#ここからDCGANモデル生成


#ニューラルネットワークとして振る舞うために必要な機能を全部まとめた親クラスがnn.Module
#それをgeneratorクラスで継承している.
class generator(nn.Module):
    #初期化
    def __init__(self):
        super(generator, self).__init__()
        #ネットワーク全体の定義
        self.net = nn.Sequential(
            #ConvTranspose2dは特徴マップを空間的に「拡大」する畳み込み.
            # 100チャネル・1×1 の入力を、4×4 のカーネルを使って、512チャネル・4×4 の特徴マップに変換する層
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),  
            #転置畳み込みで作った特徴マップ(512チャネル)を正規化(BatchNorm2d)してから非線形変換(ReLU関数)する.これにより表現力が増す.
            nn.BatchNorm2d(512), nn.ReLU(True),
            #512チャネル・4×4 の特徴マップを入力として，4×4 カーネル・ストライド2・パディング1の転置畳み込みにより，256チャネル・8×8 の特徴マップへ拡大する層
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),  
            #ここは256チャネルの特徴マップの正規化と非線形化
            nn.BatchNorm2d(256), nn.ReLU(True),
            #256チャネル・8×8 の特徴マップを入力として，転置畳み込みにより空間サイズを 16×16 に拡大し，チャネル数を 128 に減らす層
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False), 
            #ここは128チャネルの特徴マップの正規化と非線形化
            nn.BatchNorm2d(128), nn.ReLU(True),
            #128チャネル・16×16 の特徴マップを入力として，転置畳み込みにより空間サイズを 32×32 に拡大し，RGB画像に対応する 3 チャネルの出力を生成する層
            nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False),   
            #出力値を −1 から 1 の範囲に押し込める活性化関数.これを使うことで画像を[-1~1]で変換できる.
            #オライリーのDeepLearning1に書いてあった気がする.
            nn.Tanh()
        )
    '''補足
    nn.ConvTranspose2d(
    入力チャネル数,
    出力チャネル数,
    カーネルサイズ,
    ストライド,
    パディング
)

    '''
    #Generator の定義した全層を z に適用して、生成画像を返す
    def forward(self, z):
        return self.net(z)

#ディスクリミネーター(識別器)
#ジェネレーター同様,nn.Moduleで必要なモジュールを呼び出す.
class discriminator(nn.Module):
    #__init__で初期化
    def __init__(self):
        super(discriminator, self).__init__()
        #ネットワーク全体の定義
        self.net = nn.Sequential(
            #RGB画像（3チャネル・32×32）を入力として，4×4 カーネル・ストライド2・パディング1の畳み込みにより，128チャネル・16×16 の特徴マップへ変換する層
            nn.Conv2d(3, 128, 4, 2, 1, bias=False), 
            #LeakyReLUは負の値も少しだけ通す ReLU 系の活性化関数，負の領域でも小さな勾配を流し続けるため,GANの学習が安定しやすくなる.
            nn.LeakyReLU(0.2, inplace=True),
            #128チャネル・16×16 の特徴マップを入力として，4×4 カーネル・ストライド2の畳み込みにより256チャネル・8×8 の特徴マップへ変換する層
            nn.Conv2d(128, 256, 4, 2, 1, bias=False), 
            #畳み込みで抽出した 256 チャネルの特徴を正規化し，その後 LeakyReLU によって勾配を保ったまま非線形変換する
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            #256チャネル・8×8 の特徴マップを入力として，4×4 カーネル・ストライド2の畳み込みにより，512チャネル・4×4 の高次特徴マップへ変換する層.特徴を凝縮している.
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),  
            ##畳み込みで抽出した 512 チャネルの特徴を正規化し，その後 LeakyReLU によって勾配を保ったまま非線形変換する
            nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True),
            #512チャネル・4×4 にまで凝縮された特徴を，1つの値（本物らしさ）に潰すための最終層
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),    
            #活性化関数であるSigmoid関数を用いて識別器の出力を「確率」として扱える形に変換する.確立分布に落とし込む
            nn.Sigmoid()
        )
        #discriminator の定義した全層を x に適用して、生成画像を返す
    def forward(self, x):
        return self.net(x)
#で,上で作ったgeneratorクラスとdiscriminatorクラスを呼び出してPCのデバイスに入れる.
G = generator().to(device)
D = discriminator().to(device)

# 損失関数と最適化 
#BCELは2値交差エントロピー.シグモイド関数を通して予測値が[0~1]になっているため,それを正解1,不正解0と前提を置いてどれだけ正解からズレているかを示す.
criterion = nn.BCELoss()
#学習率
lr = 2e-4
#AdamWをGANで用いると逆に悪影響が出る可能性がある.最適化したいわけではなく,評価なのでAdamの方がいい.
#生成器と識別器を、それぞれ独立した Adam 最適化手法で更新するために、学習可能なパラメータを別々に登録している.
#betas = (β1, β2),β1:過去の勾配をどれだけ覚えておくか, β2:勾配の大きさのばらつきをどれだけ覚えるか
G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5,0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5,0.999))

# 学習
#エポック数
epochs = 50


#エポック数が増えるにつれて生成画像がどう変わったかを見る必要があるため,
#公平性の観点からノイズを毎回同じにする必要がある.その指定.
fixed_z = torch.randn(1, 100, 1, 1, device=device)


# エポックごとの平均損失を保存する。
#SVHNはデータ数がおおく、１エポックの中にミニバッチが12個くらいあるのでミニバッチごとにログを出すと
#見づらいので平均化する機能を実装したい
D_losses = []
G_losses = []




#エポック周回
for epoch in range(epochs):
    #1エポックの中で出てくる loss を全部集めるための一時リスト
    d_loss_epoch = []
    g_loss_epoch = []

    #DataLoaderからミニバッチ(imgs)を順番に取得する.
    #(imgs,_)の_はクラスが入るがGANでは使わないので捨てている.
    for i, (imgs, _) in enumerate(train_loader):
        #ミニバッチに含まれる画像枚数を動的に取得し，最後のバッチサイズが異なる場合でも生成画像数と一致するようにしている。
        bsz = imgs.size(0)

        # 本物画像をデバイス(CPU)に突っ込む
        real = imgs.to(device)
        #ノイズ z を生成して G に入力し、偽物画像 fake を生成
        z = torch.randn(bsz, 100, 1, 1, device=device)
        #.detach()でD(ディスクリミネーター) の学習中に G(ジェネレータ) まで更新されないようにする（Gへの逆伝播を遮断）
        fake = G(z).detach()                    
        #real_out:本物を見たときの D の出力,fake_out:偽物を見たときの D の出力
        real_out, fake_out = D(real), D(fake)
        #本物は1、偽物は0になるように D を訓練する損失
        D_loss = criterion(torch.cat([real_out, fake_out], 0),
                           torch.cat([torch.ones_like(real_out),
                                      torch.zeros_like(fake_out)], 0))
        #前回の勾配を消す
        D_optimizer.zero_grad()
        #損失から勾配を計算(誤差逆伝播方)
        D_loss.backward()
        #.stepで勾配に基づいて D の重みを更新
        D_optimizer.step()

        # 生成機をどうやって賢くするかの部分
        #標準正規分布 N(0,1) の乱数でノイズ z を生成
        z = torch.randn(bsz, 100, 1, 1, device=device)
        #ノイズｚからfake画像を生成
        fake = G(z)
        #識別器がfake画像を見た時の本物っぽさの確率
        output = D(fake)
        #Dが出した判定(output)をtorch.ones_like()で本物だと捉えている.
        G_loss = criterion(output, torch.ones_like(output))
        #前回の購買を除去
        G_optimizer.zero_grad()
        #誤差逆伝播法
        G_loss.backward()
        #重みの更新
        G_optimizer.step()
        #追加:いま処理したミニバッチ1回分の D/G の損失値を、エポック専用リストに保存する
        d_loss_epoch.append(D_loss.item())
        g_loss_epoch.append(G_loss.item())

        #100ミニバッチごとに，現在のエポック番号(Epoch)と識別器・生成器の損失(D_loss,G_los)を表示して学習状況を確認する
        #変更。ミニバッチごとにログを出力すると見づらいのでコメントアウトする.
        #if i % 100 == 0:
        #   print(f"Epoch {epoch:02d}|D_loss: {D_loss.item():.3f} | G_loss: {G_loss.item():.3f}")


#ここから下はGANとは関係ない.
        #追加：ここでmeanを用いてDとGの損失のミニバッチデータを平均化してその結果を193行目あたりにあるD_kisses,G_lossesにいれている
    D_losses.append(float(np.mean(d_loss_epoch)))
    G_losses.append(float(np.mean(g_loss_epoch)))
    print(f"[Epoch {epoch:02d}] D_loss(mean): {D_losses[-1]:.3f} | G_loss(mean): {G_losses[-1]:.3f}")

    # ===== エポックごとにサンプル画像を保存 =====
    #このブロック内では勾配計算を行わない
    with torch.no_grad():
        #同一ノイズ(fixed_z)に対する生成結果を毎エポック取得し，学習の変化を公平に比較できるようにしている
        fake_imgs = G(fixed_z).cpu()
    #出力先ディレクトリ out_dir と，エポック番号を含むファイル名を結合して，生成画像の保存パスを作っている
    save_path = os.path.join(out_dir, f"epoch_{epoch:03d}.png")
    #fake_imgsはGの活性化関数で値域が[-1~1].それを保存先のパス(save_path)に出力,正規化する.
    save_image(fake_imgs, save_path, normalize=True, value_range=(-1, 1))

    #ターミナルへの出力
    print(f"save: {save_path}")

print("\n 学習完了")
#ログファイルを閉じる.
logfile.close()