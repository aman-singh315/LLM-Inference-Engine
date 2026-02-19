# BlockPool of Paged kv
class BlockPool:
  def __init__(self, total_blocks : int, block_size : int, num_layers : int, num_heads : int , head_dim : int , device):
    self.total_blocks = total_blocks
    self.block_size = block_size
    self.device = device

    self.free_blocks = list(range(total_blocks))


    # Kv storage
    self.keys = torch.zeros(
        num_layers,
        total_blocks,
        num_heads,
        block_size,
        head_dim,
        device=device,
        dtype=torch.float16

    )

    self.values = torch.zeros(
        num_layers,
        total_blocks,
        num_heads,
        block_size,
        head_dim,
        device=device,
        dtype= torch.float16
    )

  def allocate(self):
    if not self.free_blocks:
      raise RuntimeError("Out of kv memeory blocks")

    block_id = self.free_blocks.pop()
    print(f'[Alloc] Block {block_id}')
    return block_id

  def free(self, block_id):
    self.free_blocks.append(block_id)
    print(f'[Free] Block {block_id}')
