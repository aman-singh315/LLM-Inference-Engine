# Step - 3 Request Queue
from collections import deque

class Scheduler:
  def __init__(self , max_active):
    self.waiting = deque()
    self.active = []
    self.max_active = max_active
    self.completed = []

  def Submit(self , req):
    self.waiting.append(req)

  def Inject_if_possible(self):
    while self.waiting and len(self.active) < self.max_active:
      req = self.waiting.popleft()
      self.active.append(req)
