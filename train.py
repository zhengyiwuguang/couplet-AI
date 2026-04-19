import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import re
from tqdm import tqdm

# ==================== 最终优化配置 ====================
MAX_LEN = 16
BATCH_SIZE = 32
EMBED_DIM = 128
FF_DIM = 256
HEAD_NUM = 4
ENC_LAYER = 1
DEC_LAYER = 1
LR = 1e-4
EPOCHS = 10  # 增加训练轮数，效果更好
DEVICE = torch.device('cpu')
# =========================================================

def clean_text(s):
    s = re.compile(r'[^\u4e00-\u9fa5]').sub('', s)
    return s[:MAX_LEN]

class CoupletDataset(Dataset):
    def __init__(self, up_path="up.txt", down_path="down.txt"):
        print("正在加载上联...")
        with open(up_path, 'r', encoding='utf-8', errors='ignore') as f:
            self.up = [clean_text(line.strip()) for line in tqdm(f) if line.strip()]

        print("正在加载下联...")
        with open(down_path, 'r', encoding='utf-8', errors='ignore') as f:
            self.down = [clean_text(line.strip()) for line in tqdm(f) if line.strip()]

        # 1. 取数量最小的
        n = min(len(self.up), len(self.down))
        self.up = self.up[:n]
        self.down = self.down[:n]

        # 2. 【核心修复】强制保留 字数完全相等 的对联
        filtered_up = []
        filtered_down = []
        for u, d in zip(self.up, self.down):
            if len(u) == len(d) and len(u) > 0:
                filtered_up.append(u)
                filtered_down.append(d)
        self.up = filtered_up
        self.down = filtered_down

        print(f"有效对联数量（字数严格对齐）：{len(self.up)} 副")
        self.build_vocab()

    def build_vocab(self):
        chars = set()
        for u, d in tqdm(zip(self.up, self.down), total=len(self.up)):
            chars.update(u)
            chars.update(d)

        self.vocab = {"<PAD>":0, "<SOS>":1, "<EOS>":2, "<UNK>":3}
        for c in sorted(chars):
            self.vocab[c] = len(self.vocab)

        self.idx2word = {v:k for k,v in self.vocab.items()}
        pickle.dump(self.vocab, open("vocab.pkl", "wb"))
        print(f"词汇表大小：{len(self.vocab)}")

    def __len__(self):
        return len(self.up)

    def encode(self, s):
        idx = [self.vocab.get(c,3) for c in s]
        idx.append(2)
        while len(idx) < MAX_LEN:
            idx.append(0)
        return torch.tensor(idx[:MAX_LEN])

    def __getitem__(self, i):
        return self.encode(self.up[i]), self.encode(self.down[i])

# Transformer 模型
class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2) * (-np.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:,0::2] = torch.sin(pos * div)
        pe[:,1::2] = torch.cos(pos * div)
        self.pe = pe.unsqueeze(0)
    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

class TransformerModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, EMBED_DIM)
        self.pe = PositionalEncoding(EMBED_DIM)
        self.tfm = nn.Transformer(
            d_model=EMBED_DIM, nhead=HEAD_NUM,
            num_encoder_layers=ENC_LAYER, num_decoder_layers=DEC_LAYER,
            dim_feedforward=FF_DIM, batch_first=True
        )
        self.fc = nn.Linear(EMBED_DIM, vocab_size)

    def forward(self, src, tgt):
        # 【核心修复】Encoder 输入 绝对不能加 mask！
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(DEVICE)
        
        src = self.pe(self.emb(src))
        tgt = self.pe(self.emb(tgt))
        
        # 只传入 tgt_mask
        out = self.tfm(src, tgt, tgt_mask=tgt_mask)
        return self.fc(out)

if __name__ == "__main__":
    dataset = CoupletDataset()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    model = TransformerModel(len(dataset.vocab)).to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print("\n开始训练...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for src, tgt in pbar:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            optimizer.zero_grad()
            
            pred = model(src, tgt[:, :-1])
            loss = criterion(pred.reshape(-1, pred.size(-1)), tgt[:, 1:].reshape(-1))
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        print(f"==> Epoch {epoch+1} 平均损失: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "couplet_model.pth")
    print("\n训练完成！模型已保存")