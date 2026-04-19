import torch
import pickle
from train import TransformerModel, MAX_LEN, DEVICE, clean_text

vocab = pickle.load(open("vocab.pkl", "rb"))
idx2word = {v: k for k, v in vocab.items()}
model = TransformerModel(len(vocab)).to(DEVICE)
model.load_state_dict(torch.load("couplet_model.pth", map_location=DEVICE))
model.eval()

def generate_couplet(up_text):
    up_text = clean_text(up_text)
    up_len = len(up_text)

    src = [vocab.get(c, 3) for c in up_text]
    src.append(2)
    while len(src) < MAX_LEN:
        src.append(0)
    src = torch.tensor([src]).to(DEVICE)

    tgt = [vocab["<SOS>"]]
    with torch.no_grad():
        for _ in range(up_len):
            tgt_tensor = torch.tensor([tgt]).to(DEVICE)
            out = model(src, tgt_tensor)
            next_idx = out.argmax(-1)[:, -1].item()
            if next_idx in (2, 3):
                next_idx = 4
            tgt.append(next_idx)

    return "".join([idx2word[i] for i in tgt[1:up_len+1]])

if __name__ == "__main__":
    print("=== 智能对联生成系统 ===")
    while True:
        up = input("请输入上联：")
        if up in ["exit", "quit", "q"]:
            break
        down = generate_couplet(up)
        print(f"下联：{down}\n")