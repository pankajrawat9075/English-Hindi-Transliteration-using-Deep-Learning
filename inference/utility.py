import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder_block_size = 33
decoder_block_size = 30

class Head(nn.Module):
    """ one self-attention head """

    def __init__(self, n_embd, d_k, dropout, mask=0): # d_k is dimention of key , nomaly d_k = n_embd / 4
        super().__init__()
        self.mask = mask
        self.key = nn.Linear(n_embd, d_k, bias=False, device=device)
        self.query = nn.Linear(n_embd, d_k, bias=False, device=device)
        self.value = nn.Linear(n_embd, d_k, bias=False, device=device)
        if mask:
            self.register_buffer('tril', torch.tril(torch.ones(encoder_block_size, encoder_block_size, device=device)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output = None):
        B,T,C = x.shape

        if encoder_output is not None:
            k = self.key(encoder_output)
            Be, Te, Ce = encoder_output.shape
        else:
            k = self.key(x) # (B,T,d_k)

        q = self.query(x) # (B,T,d_k)
        # compute attention scores
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B,T,T)

        if self.mask:
            if encoder_output is not None:
                wei = wei.masked_fill(self.tril[:T, :Te] == 0, float('-inf')) # (B,T,T)
            else:
                wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B,T,T)

        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        # perform weighted aggregation of values
        if encoder_output is not None:
            v = self.value(encoder_output)
        else:
            v = self.value(x)
        out = wei @ v # (B,T,C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple self attention heads in parallel """

    def __init__(self, n_embd, num_head, d_k, dropout, mask=0):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, d_k, dropout, mask) for _ in range(num_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output=None):
        out = torch.cat([h(x, encoder_output) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ multiple self attention heads in parallel """

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class encoderBlock(nn.Module):
    """ Tranformer encoder block : communication followed by computation """

    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        d_k = n_embd // n_head
        self.sa = MultiHeadAttention(n_embd, n_head, d_k, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, encoder_output=None):
        x = x + self.sa(self.ln1(x), encoder_output)
        x = x + self.ffwd(self.ln2(x))
        return x

class Encoder(nn.Module):

    def __init__(self, n_embd, n_head, n_layers, dropout):
        super().__init__()

        self.token_embedding_table = nn.Embedding(input_vocab_size, n_embd) # n_embd: input embedding dimension
        self.position_embedding_table = nn.Embedding(encoder_block_size, n_embd)
        self.blocks = nn.Sequential(*[encoderBlock(n_embd, n_head, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B,T,n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,n_embd)
        x = tok_emb + pos_emb # (B,T,n_embd)
        x = self.blocks(x) # apply one attention layer (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        return x


class decoderBlock(nn.Module):
    """ Tranformer decoder block : self communication then cross communication followed by computation """

    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        d_k = n_embd // n_head
        self.sa = MultiHeadAttention(n_embd, n_head, d_k, dropout, mask = 1)
        self.ca = MultiHeadAttention(n_embd, n_head, d_k, dropout, mask = 1)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd, device=device)
        self.ln2 = nn.LayerNorm(n_embd, device=device)
        self.ln3 = nn.LayerNorm(n_embd, device=device)

    def forward(self, x_encoder_output):
        x = x_encoder_output[0]
        encoder_output = x_encoder_output[1]
        x = x + self.sa(self.ln1(x))
        x = x + self.ca(self.ln2(x), encoder_output)
        x = x + self.ffwd(self.ln3(x))
        return (x,encoder_output)

class Decoder(nn.Module):

    def __init__(self, n_embd, n_head, n_layers, dropout):
        super().__init__()

        self.token_embedding_table = nn.Embedding(output_vocab_size, n_embd) # n_embd: input embedding dimension
        self.position_embedding_table = nn.Embedding(decoder_block_size, n_embd)
        self.blocks = nn.Sequential(*[decoderBlock(n_embd, n_head=n_head, dropout=dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, output_vocab_size)

    def forward(self, idx, encoder_output, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) # (B,T,n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,n_embd)
        x = tok_emb + pos_emb # (B,T,n_embd)

        x =self.blocks((x, encoder_output))
        x = self.ln_f(x[0]) # (B,T,C)
        logits = self.lm_head(x) # (B,T,output_vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            temp_logits = logits.view(B*T, C)
            targets = targets.reshape(B*T)

            loss = F.cross_entropy(temp_logits, targets.long())

        # print(logits)
        # out = torch.argmax(logits)

        return logits, loss

