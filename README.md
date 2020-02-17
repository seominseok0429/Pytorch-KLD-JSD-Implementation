# Pytorch-KLD-JSD-Implementation

***
### Kullback-Leibler divergence

```python
import torch
import torch.nn.functional as F

def KLD_loss(outputs, labels, teacher_outputs):
    alpha = 0.9
    T = 5 # Temperature
    outputs = F.softmax(outputs/T, dim=1)
    outputs = torch.log(outputs)
    teacher_outputs = F.softmax(teacher_outputs/T, dim=1)
    KD_loss = nn.KLDivLoss()(outputs, teacher_outputs) * (alpha * T * T) + /
                             F.cross_entropy(outputs, labels) * (1. - alpha)
    
    return KD_loss
```
***
###  Jessen-Shannon Divergence

```python
import torch
import torch.nn.functional as F

def JSD_loss(outputs1, labels, outputs2):
    alpha = 0.9
    T = 5
    outputs1 = F.softmax(outputs1/T, dim=1)
    outputs1 = torch.log(outputs1)
    outputs2 = F.softmax(outputs2/T, dim=1)
    outputs2 = torch.log(outputs2)
    M = 0.5*(F.softmax(teacher_outputs/T, dim=1)+F.softmax(outputs/T, dim=1))
    KD_loss = 0.5*(nn.KLDivLoss()(outputs1,M) +  nn.KLDivLoss()(outputs2,M)) * (alpha * T * T) + \
              0.5*(F.cross_entropy(outputs1, labels)+F.cross_entropy(outputs2, labels)) * (1. - alpha)
    return KD_loss
```
***

### simple implementation

```python
   for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs1 = net1(inputs)
        outputs2 = net2(inputs)
        loss = KLD_loss(outputs1, targets, outputs2)
        # loss = JSD_loss(outputs1, targets, outputs2)
        loss.backward()
```
## sharpen

```python
# input = self.softmax(x)
def sharpen(x, T):
    temp = x**(1/T)
    return temp / temp.sum(axis=1, keepdims=True)

```
