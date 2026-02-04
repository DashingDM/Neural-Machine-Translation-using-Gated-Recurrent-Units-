
import os, sys, random
from pathlib import Path
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from tqdm.notebook import tqdm
import warnings
warnings.filterwarnings('ignore')

try:
    from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
    import nltk
    nltk.download('punkt', quiet=True)
except:
    os.system('pip install -q nltk')
    from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
    import nltk
    nltk.download('punkt', quiet=True)

try:
    import google.colab
    IN_COLAB = True
except:
    IN_COLAB = False
    from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    


def find_dataset():
    search_paths = ["data/en-vi", "data", "en-vi", ".", "/content/data/en-vi", "/content/data", "/content"]

    for base in search_paths:
        en = Path(base) / "train.en"
        vi = Path(base) / "train.vi"
        if en.exists() and vi.exists():
            test_en = Path(base) / "tst2013.en"
            test_vi = Path(base) / "tst2013.vi"
            if not test_en.exists():
                test_en = Path(base) / "tst2012.en"
                test_vi = Path(base) / "tst2012.vi"
            
            return str(en), str(vi), str(test_en), str(test_vi)

    raise FileNotFoundError("Dataset not found.")

def build_vocab(sentences, min_freq=2):
    
    counter = Counter()
    for sentence in sentences:
        counter.update(sentence.strip().lower().split())
    vocab = {"<pad>":0, "<unk>":1, "<sos>":2, "<eos>":3}
    idx = 4
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = idx
            idx += 1
    return vocab

class TranslationDataset(Dataset):
    def __init__(self, src_list, tgt_list, min_freq=2, max_len=40):
        
        pairs = [(s.strip().lower(), t.strip().lower())
                 for s, t in zip(src_list, tgt_list)
                 if s.strip() and t.strip()]
        pairs = [(s, t) for s, t in pairs
                 if len(s.split()) <= max_len and len(t.split()) <= max_len]

        self.src = [p[0] for p in pairs]
        self.tgt = [p[1] for p in pairs]

        self.src_vocab = build_vocab(self.src, min_freq)
        self.tgt_vocab = build_vocab(self.tgt, min_freq)
        self.tgt_ivocab = {v:k for k,v in self.tgt_vocab.items()}

       

    def __len__(self):
        return len(self.src)

    def encode(self, sentence, vocab):
        return [vocab.get(w, vocab["<unk>"]) for w in sentence.strip().split()]

    def __getitem__(self, idx):
        src = [self.src_vocab.get(w, self.src_vocab["<unk>"]) for w in self.src[idx].split()]
        tgt = [self.tgt_vocab["<sos>"]] + \
              [self.tgt_vocab.get(w, self.tgt_vocab["<unk>"]) for w in self.tgt[idx].split()] + \
              [self.tgt_vocab["<eos>"]]
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)

def collate_fn(batch, src_pad, tgt_pad):
    src_batch, tgt_batch = zip(*batch)
    src_batch = nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=src_pad)
    tgt_batch = nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=tgt_pad)
    return src_batch, tgt_batch

def get_teacher_forcing_ratio(epoch, total_epochs=15, start=1.0, end=0.4):
    
    if epoch <= 2:
        return start
    progress = (epoch - 2) / (total_epochs - 2)
    return start * (end / start) ** progress


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        
        self.rnn = nn.GRU(embed_size, hidden_size, num_layers=num_layers,
                          batch_first=True, dropout=0, bidirectional=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        outputs, hidden = self.rnn(embedded)
        return outputs, hidden

class LuongAttention(nn.Module):
    
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden, encoder_outputs):
        
        hidden = hidden[-1].unsqueeze(1)  
        energy = torch.bmm(hidden, encoder_outputs.transpose(1, 2))  
        return F.softmax(energy.squeeze(1), dim=1)

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.attention = LuongAttention(hidden_size)
        self.rnn = nn.GRU(embed_size + hidden_size, hidden_size,
                          num_layers=num_layers, batch_first=True, dropout=0)
        
        self.fc = nn.Linear(hidden_size * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden, encoder_outputs):
        x = x.unsqueeze(1)
        embedded = self.dropout(self.embedding(x))

        attn_weights = self.attention(hidden, encoder_outputs).unsqueeze(1)
        context = torch.bmm(attn_weights, encoder_outputs)

        rnn_input = torch.cat([embedded, context], dim=2)
        output, hidden = self.rnn(rnn_input, hidden)

        output = torch.cat([output, context], dim=2)
        pred = self.fc(output.squeeze(1))
        return pred, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size, tgt_len = tgt.size()
        vocab_size = self.decoder.fc.out_features
        outputs = torch.zeros(batch_size, tgt_len-1, vocab_size, device=device)

        encoder_outputs, hidden = self.encoder(src)
        input_tok = tgt[:, 0]

        for t in range(1, tgt_len):
            output, hidden = self.decoder(input_tok, hidden, encoder_outputs)
            outputs[:, t-1, :] = output

            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input_tok = tgt[:, t] if teacher_force else top1

        return outputs


def calculate_bleu(model, val_data, train_data, num_samples=30, max_len=40):
    model.eval()
    bleu_scores = []
    smoothie = SmoothingFunction().method4
    num_samples = min(len(val_data), num_samples)

    for i in range(num_samples):
        src_sentence = val_data.src[i]
        tgt_sentence = val_data.tgt[i]

        src_tensor = torch.tensor(
            [val_data.encode(src_sentence, train_data.src_vocab)],
            dtype=torch.long
        ).to(device)

        with torch.no_grad():
            encoder_outputs, hidden = model.encoder(src_tensor)
            dec_input = torch.tensor([train_data.tgt_vocab["<sos>"]], device=device)
            pred_ids = []

            for _ in range(max_len):
                output, hidden = model.decoder(dec_input, hidden, encoder_outputs)
                wid = output.argmax(1).item()
                if wid == train_data.tgt_vocab["<eos>"]:
                    break
                pred_ids.append(wid)
                dec_input = torch.tensor([wid], device=device)

        pred_words = [train_data.tgt_ivocab.get(i, "<unk>") for i in pred_ids]
        ref_words = tgt_sentence.split()

        if pred_words:
            bleu = sentence_bleu([ref_words], pred_words, smoothing_function=smoothie)
            bleu_scores.append(bleu)

    return sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0

def train(epochs=30, batch_size=32, lr=0.0005, max_pairs=20000,
          hidden_size=512, embed_size=256, num_layers=2, dropout=0.3,
          patience=10, min_freq=1, validate_every=1):

    train_en, train_vi, val_en, val_vi = find_dataset()

    with open(train_en, "r", encoding="utf-8") as f:
        en_lines = f.readlines()
    with open(train_vi, "r", encoding="utf-8") as f:
        vi_lines = f.readlines()

    if len(en_lines) > max_pairs:
        en_lines, vi_lines = en_lines[:max_pairs], vi_lines[:max_pairs]

    data = TranslationDataset(en_lines, vi_lines, min_freq=min_freq, max_len=40)

    loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,  
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=lambda b: collate_fn(b, data.src_vocab["<pad>"], data.tgt_vocab["<pad>"])
    )

    
    val_en_lines, val_vi_lines = [], []
    if Path(val_en).exists() and Path(val_vi).exists():
        with open(val_en, "r", encoding="utf-8") as f:
            val_en_lines = f.readlines()
        with open(val_vi, "r", encoding="utf-8") as f:
            val_vi_lines = f.readlines()

    
    model = Seq2Seq(
        Encoder(len(data.src_vocab), embed_size, hidden_size, num_layers, dropout),
        Decoder(len(data.tgt_vocab), embed_size, hidden_size, num_layers, dropout)
    ).to(device)

   

    opt = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5,
                                                      patience=2, verbose=True)

    loss_fn = nn.CrossEntropyLoss(ignore_index=data.tgt_vocab["<pad>"])

    best_bleu = 0.0
    epochs_no_improve = 0
    os.makedirs("model", exist_ok=True)

    import time
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        tf_ratio = get_teacher_forcing_ratio(epoch, epochs)

        epoch_start = time.time()
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}")

        for src, tgt in pbar:
            src, tgt = src.to(device, non_blocking=True), tgt.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)  
            out = model(src, tgt, tf_ratio)
            o = out.reshape(-1, out.shape[-1])
            t = tgt[:, 1:].reshape(-1)

            loss = loss_fn(o, t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += loss.item()
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "tf": f"{tf_ratio:.3f}"
            })

        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | TF: {tf_ratio:.3f} | Time: {epoch_time:.1f}s")

        
        if val_en_lines and epoch % validate_every == 0:
            val_data = TranslationDataset(val_en_lines, val_vi_lines, min_freq=min_freq)
            avg_bleu = calculate_bleu(model, val_data, data, num_samples=30)
            

            scheduler.step(avg_bleu)

            if avg_bleu > best_bleu:
                best_bleu = avg_bleu
                epochs_no_improve = 0
                torch.save({
                    "model": model.state_dict(),
                    "src_vocab": data.src_vocab,
                    "tgt_vocab": data.tgt_vocab,
                    "epoch": epoch,
                    "bleu": avg_bleu,
                    "config": {
                        "hidden_size": hidden_size,
                        "embed_size": embed_size,
                        "num_layers": num_layers,
                        "dropout": dropout
                    }
                }, "model/best_model.pt")
                
            else:
                epochs_no_improve += 1
                

            if epochs_no_improve >= patience:
                break
        print()

    total_time = time.time() - start_time
    
    print(f"model")
    return model

def test(test_file_en=None, test_file_vi=None, num_samples=None):
    

    checkpoint = torch.load("model/best_model.pt", map_location=device)
    src_vocab = checkpoint["src_vocab"]
    tgt_vocab = checkpoint["tgt_vocab"]
    tgt_ivocab = {v:k for k,v in tgt_vocab.items()}

    config = checkpoint.get("config", {
        "hidden_size": 256, "embed_size": 128, "num_layers": 1, "dropout": 0.2
    })

    model = Seq2Seq(
        Encoder(len(src_vocab), config["embed_size"], config["hidden_size"],
                config["num_layers"], config["dropout"]),
        Decoder(len(tgt_vocab), config["embed_size"], config["hidden_size"],
                config["num_layers"], config["dropout"])
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    

    if test_file_en is None or test_file_vi is None:
        _, _, test_file_en, test_file_vi = find_dataset()

    with open(test_file_en, "r", encoding="utf-8") as f:
        test_en = [line.strip().lower() for line in f.readlines() if line.strip()]
    with open(test_file_vi, "r", encoding="utf-8") as f:
        test_vi = [line.strip().lower() for line in f.readlines() if line.strip()]

    if num_samples:
        test_en = test_en[:num_samples]
        test_vi = test_vi[:num_samples]

    
    predictions = []
    references = []
    smoothie = SmoothingFunction().method4
    bleu_scores = []

    for src_sentence, ref_sentence in tqdm(zip(test_en, test_vi), total=len(test_en)):
        ids = [src_vocab.get(w, src_vocab["<unk>"]) for w in src_sentence.split()]
        if not ids:
            continue

        src = torch.tensor([ids], device=device)

        with torch.no_grad():
            encoder_outputs, hidden = model.encoder(src)
            dec_input = torch.tensor([tgt_vocab["<sos>"]], device=device)
            pred_ids = []

            for _ in range(40):
                output, hidden = model.decoder(dec_input, hidden, encoder_outputs)
                wid = output.argmax(1).item()
                if wid == tgt_vocab["<eos>"]:
                    break
                pred_ids.append(wid)
                dec_input = torch.tensor([wid], device=device)

        pred_sentence = " ".join([tgt_ivocab.get(idx, "<unk>") for idx in pred_ids])
        if len(predictions) < 20:  
            print(pred_sentence)
        predictions.append(pred_sentence.split())
        references.append([ref_sentence.split()])

        if pred_sentence.split():
            bleu = sentence_bleu([ref_sentence.split()], pred_sentence.split(),
                               smoothing_function=smoothie)
            bleu_scores.append(bleu)

    corpus_bleu_score = corpus_bleu(references, predictions, smoothing_function=smoothie)
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0

    
    print(f"\nCorpus BLEU: {corpus_bleu_score:.4f}")
    print(f"Avg sentence BLEU: {avg_bleu:.4f}")
   

    return corpus_bleu_score





if __name__ == "__main__":
    if len(sys.argv) < 2:
        
        sys.exit()
    
    mode = sys.argv[1]
    
    if mode == "train":
        train()
    elif mode == "test":
        test()
   
    
