# Step - 1 Request Object
class Request:
  def __init__(self , req_id , prompt, max_new_tokens):
    self.req_id = req_id

    # Prompt
    self.prompt = prompt

    # generation control
    self.max_new_tokens = max_new_tokens
    self.num_generated = 0
    self.finished = False

    self.state =  None

    self.slot_id = None
    self.seq_len = 0
    self.last_tokens = None
    self.output_tokens = []

# Step - 2
class RequestState:
  def __init__(self):
    self.block_ids = []
    self.total_tokens = 0
