
import os
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_OK = True
except Exception:  # kapta torch yoksa
    TORCH_OK = False


if TORCH_OK:
    class Encoder1D(nn.Module):

        def __init__(self, in_ch, width=64, embed_dim=128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(in_ch, width, 7, stride=2, padding=3), nn.BatchNorm1d(width), nn.GELU(),
                nn.Conv1d(width, width * 2, 5, stride=2, padding=2), nn.BatchNorm1d(width * 2), nn.GELU(),
                nn.Conv1d(width * 2, width * 2, 5, stride=2, padding=2), nn.BatchNorm1d(width * 2), nn.GELU(),
                nn.Conv1d(width * 2, embed_dim, 3, stride=2, padding=1), nn.BatchNorm1d(embed_dim), nn.GELU(),
            )
            self.embed_dim = embed_dim

        def forward(self, x):              # x: (B, C, T)
            h = self.net(x)                # (B, E, T')
            return h

        def embed(self, x):
            h = self.forward(x)
            return h.mean(dim=-1)          # (B, E) global ortalama-pool

    class ReconDecoder(nn.Module):

        def __init__(self, in_ch, width=64, embed_dim=128):
            super().__init__()
            self.net = nn.Sequential(
                nn.ConvTranspose1d(embed_dim, width * 2, 4, stride=2, padding=1), nn.GELU(),
                nn.ConvTranspose1d(width * 2, width * 2, 4, stride=2, padding=1), nn.GELU(),
                nn.ConvTranspose1d(width * 2, width, 4, stride=2, padding=1), nn.GELU(),
                nn.ConvTranspose1d(width, in_ch, 4, stride=2, padding=1),
            )

        def forward(self, h):
            return self.net(h)

    class SSLModel(nn.Module):
        def __init__(self, in_ch, width, embed_dim):
            super().__init__()
            self.enc = Encoder1D(in_ch, width, embed_dim)
            self.dec = ReconDecoder(in_ch, width, embed_dim)

        def forward(self, x):
            return self.dec(self.enc(x))

    class Classifier(nn.Module):
        def __init__(self, encoder, embed_dim, freeze_encoder=False):
            super().__init__()
            self.enc = encoder
            self.freeze = freeze_encoder
            self.head = nn.Sequential(nn.Linear(embed_dim, 64), nn.GELU(), nn.Linear(64, 1))

        def forward(self, x):
            if self.freeze:
                with torch.no_grad():
                    z = self.enc.embed(x)
            else:
                z = self.enc.embed(x)
            return self.head(z).squeeze(-1)   # logit



    def random_mask(x, ratio):


        B, C, T = x.shape
        mask = torch.ones(B, 1, T, device=x.device)
        n_mask = int(T * ratio)
        for b in range(B):
            idx = torch.randperm(T, device=x.device)[:n_mask]
            mask[b, 0, idx] = 0.0
        return x * mask, (1 - mask)  # masked_input, masked_region

    def pretrain_ssl(unlabeled_X, cfg, in_ch, ckpt_path, mask_ratio=None):


        mask_ratio = cfg.ssl_mask_ratio if mask_ratio is None else mask_ratio
        dev = cfg.device if torch.cuda.is_available() else "cpu"
        model = SSLModel(in_ch, cfg.encoder_width, cfg.embed_dim).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.ssl_lr, weight_decay=1e-4)
        X = torch.tensor(unlabeled_X, dtype=torch.float32)
        n = len(X)
        start_epoch = 0
        if os.path.exists(ckpt_path):  # devam-et (12 sa/oturum kotası için)
            ck = torch.load(ckpt_path, map_location=dev)
            model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
            start_epoch = ck["epoch"]
        model.train()
        for ep in range(start_epoch, cfg.ssl_epochs):
            perm = torch.randperm(n)
            tot = 0.0
            for i in range(0, n, cfg.ssl_batch_size):
                xb = X[perm[i:i + cfg.ssl_batch_size]].to(dev)
                masked, region = random_mask(xb, mask_ratio)
                rec = model(masked)
                rec = rec[..., : xb.shape[-1]]
                region = region[..., : xb.shape[-1]]
                loss = (((rec - xb) ** 2) * region).sum() / (region.sum() + 1e-8)
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item() * len(xb)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "epoch": ep + 1}, ckpt_path)
        return model.enc

    def build_encoder(in_ch, cfg):
        return Encoder1D(in_ch, cfg.encoder_width, cfg.embed_dim)

    def train_classifier(enc, Xtr, ytr, cfg, freeze=False, init_from=None):
        dev = cfg.device if torch.cuda.is_available() else "cpu"
        clf = Classifier(enc, cfg.embed_dim, freeze_encoder=freeze).to(dev)
        if init_from is not None:
            clf.enc.load_state_dict(init_from.state_dict())
        opt = torch.optim.AdamW(clf.parameters(), lr=cfg.clf_lr, weight_decay=1e-4)
        lossf = nn.BCEWithLogitsLoss()
        X = torch.tensor(Xtr, dtype=torch.float32); y = torch.tensor(ytr, dtype=torch.float32)
        n = len(X)
        clf.train()
        for ep in range(cfg.finetune_epochs):
            perm = torch.randperm(n)
            for i in range(0, n, cfg.clf_batch_size):
                idx = perm[i:i + cfg.clf_batch_size]
                xb = X[idx].to(dev); yb = y[idx].to(dev)
                logit = clf(xb)
                loss = lossf(logit, yb)
                opt.zero_grad(); loss.backward(); opt.step()
        return clf

    @torch.no_grad()
    def predict_logits(clf, X, cfg):
        dev = cfg.device if torch.cuda.is_available() else "cpu"
        clf.eval()
        Xt = torch.tensor(X, dtype=torch.float32)
        out = []
        for i in range(0, len(Xt), 512):
            out.append(clf(Xt[i:i + 512].to(dev)).cpu().numpy())
        return np.concatenate(out)
