import torch

def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    #torch.triu: return the matrix of all elements on and aobove main diagonal
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
memory = torch.rand(50, 32, 128)
tgt = torch.rand(50, 32, 128)
memory = memory.to(device)
tgt = tgt.to(device)
tgt_mask=generate_square_subsequent_mask(tgt.size(0),device)
memory_key_padding_mask = None
tgt_bi_valid_mask = None

tgt = tgt.transpose(0, 1)
memory = memory.transpose(0, 1)

hidden_states = torch.cat((memory, tgt), dim=1)
num_tgt = tgt.shape[1]
num_memory = memory.shape[1]
device = tgt.device
dtype = tgt.dtype
top_left = torch.zeros((num_memory, num_memory), device=device, dtype=dtype)
top_right = torch.full((num_memory, num_tgt), float('-inf'), device=tgt.device, dtype=dtype,)
bottom_left = torch.zeros((num_tgt, num_memory), dtype=dtype, device=device,)
left = torch.cat((top_left, bottom_left), dim=0)
right = torch.cat((top_right, tgt_mask.to(dtype)), dim=0)

full_attention_mask = torch.cat((left, right), dim=1)[None, :]
print('full_attention_mask.shape: ',full_attention_mask.shape)
print('full_attention_mask: ',full_attention_mask)

if memory_key_padding_mask is None:
    memory_key_padding_mask = torch.full((memory.shape[0], memory.shape[1]), fill_value=False, device=device)
# if it is False, it means valid. That is, it is not a padding
assert memory_key_padding_mask.dtype == torch.bool
zero_negative_infinity = torch.zeros_like(memory_key_padding_mask, dtype=tgt.dtype)
zero_negative_infinity[memory_key_padding_mask] = float('-inf')
full_attention_mask = full_attention_mask.expand((memory_key_padding_mask.shape[0], num_memory + num_tgt, num_memory + num_tgt))
full_attention_mask = full_attention_mask.clone()
origin_left = full_attention_mask[:, :, :num_memory]
update = zero_negative_infinity[:, None, :]
full_attention_mask[:, :, :num_memory] = origin_left + update

if tgt_bi_valid_mask is not None:
    # verify the correctness
    bs = full_attention_mask.shape[0]
    # during inference, tgt_bi_valid_mask's length is not changed, but
    # num_tgt can be increased
    max_valid_target = tgt_bi_valid_mask.shape[1]
    mask = tgt_bi_valid_mask[:, None, :].expand((bs, num_memory+num_tgt, max_valid_target))
    full_attention_mask[:, :, num_memory:(num_memory+max_valid_target)][mask] = 0

# add axis for multi-head
full_attention_mask = full_attention_mask[:, None, :, :]