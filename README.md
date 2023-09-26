# Zeptograd

A very small deep learning framework. Operate on (numpy) tensors, choosing from a small opset. Perform autograd. Perform SGD optimization.

Like [micrograd](https://github.com/karpathy/micrograd), [nanograd](https://github.com/PABannier/nanograd), [picograd](https://github.com/breandan/picograd) or [femtograd](https://metafunctor.com/project/femtograd/), but... smaller(?).

## Install

```bash
# inside cloned repo
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Example

```python
"""An example comparing zeptograd with pytorch"""

# Initialise some tensor data
import numpy as np
t0_data = np.random.rand(2, 2)
t1_data = np.random.rand(2, 2)

# The forward graph:
#
# t0 -- Matmul -- mm -- Relu -- relu -- Sum -- loss
# t1 ---'

from zeptograd import Tensor, SGD
t0 = Tensor(t0_data)
t1 = Tensor(t1_data)
mm = t0.matmul(t1)
relu = mm.relu()
loss = relu.sum()
loss.backward()
optimizer = SGD([t0, t1], lr=0.1)
optimizer.step()

import torch
t_t0 = torch.from_numpy(t0_data)
t_t0.requires_grad = True
t_t1 = torch.from_numpy(t1_data)
t_t1.requires_grad = True
t_mm = torch.matmul(t_t0, t_t1)
t_relu = torch.relu(t_mm)
t_loss = torch.sum(t_relu)
t_loss.backward()
optimizer = torch.optim.SGD([t_t0, t_t1], lr=0.1)
optimizer.step()

assert np.allclose(t0.data, t_t0.data)
assert np.allclose(t1.data, t_t1.data)
```
