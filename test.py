import torch
import torch.nn.functional as F

import torch
from torch.autograd import Function

class CustomCrossEntropyLoss(Function):
    @staticmethod
    def forward(ctx, logits, target):
        log_softmax_output = torch.log_softmax(logits, dim=-1)
        loss = torch.mean(-log_softmax_output[range(target.shape[0]), target])

        ctx.save_for_backward(log_softmax_output, target)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        log_softmax_output, target = ctx.saved_tensors
        softmax_output = torch.exp(log_softmax_output)

        grad_input = softmax_output.clone()
        grad_input[range(target.shape[0]), target] -= 1
        grad_input /= target.shape[0]

        return grad_output * grad_input, None

def custom_cross_entropy_loss(logits, target):
    return CustomCrossEntropyLoss.apply(logits, target)

# Define a set of logits
logits = torch.tensor([[5., 1., 7.]], requires_grad=True)

# Compute the softmax probabilities
probs = custom_cross_entropy_loss(logits)

# one_hot = torch.zeros_like(probs)
# one_hot.scatter_(1, probs.argmax(dim=1, keepdim=True), 1)

labels = torch.argmax(probs, dim=1)

# Print the probabilities
print(probs)
print(labels)

# print(one_hot)

for i in range(1, 1 + 1):
    print(i)
