import torch
import struct
from load import load_checkpoint

if __name__ == "__main__":
    modelpath = "/home/cxd/Celeritas/models/stories15M.pt"
    model = load_checkpoint(modelpath)
    filepath = "/home/cxd/Celeritas/tmp/test.bin"
    out_file = open(filepath, 'wb')

    hidden_dim = model.layers[0].feed_forward.w1.weight.shape[0]
    p = model.params
    shared_classifier = torch.equal(model.tok_embeddings.weight, model.output.weight)

    if not shared_classifier:
        p.vocab_size = -p.vocab_size
    n_kv_heads = p.n_heads if p.n_kv_heads is None else p.n_kv_heads
    header = struct.pack('iiiiiii', p.dim, hidden_dim, p.n_layers, p.n_heads,
                        n_kv_heads, p.vocab_size, p.max_seq_len)
    out_file.write(header)