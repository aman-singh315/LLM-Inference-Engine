
class BlockPool:
  def __init__(self, total_blocks : int, block_size : int, num_layers : int, num_heads : int , head_dim : int , device):
    self.total_blocks = total_blocks
    self.block_size = block_size
    self.device = device

    self.free_blocks = list(range(total_blocks))   # free_blocks is simply a list of which blocks are currently available with an id.


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

  def allocate(self):   # allocate do not allocate gpu memory it just says Give this request ownership of block 4. if initialy block size is 5
    if not self.free_blocks:
      raise RuntimeError("Out of kv memeory Block")

    block_id = self.free_blocks.pop()  # here we are removing it because block containes a request kv cache
    print(f'[Alloc] Block {block_id}')
    return block_id

  def free(self, block_id):
    self.keys[:, block_id, :, :, :] = 0.0
    self.values[:, block_id, :, :, :] = 0.0
    self.free_blocks.append(block_id)  # here we are again appending it because the request has been completed so making it ready for next request
    print(f'[Free] Block {block_id}')  # even after completion of request A the same block can hold the previous kv cache but once the block is allocated to
                                       # request B it overrides the previous kv cache
