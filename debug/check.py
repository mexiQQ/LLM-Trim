import torch

random1 = torch.load("./random1.pt")
random2 = torch.load("./random2.pt")

check = torch.equal(random1, random2)

import pdb; pdb.set_trace()