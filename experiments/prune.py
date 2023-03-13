import torch
from torch.nn.utils.prune import L1Unstructured
from evaluate_after_prune import read_data
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.prune import BasePruningMethod

def train(model, loader, criterion, optimizer, device, max_norm):
    model.train()
    total_loss = 0
    for padded_text, attention_masks, labels in loader:
        padded_text = padded_text.to(device)
        attention_masks = attention_masks.to(device)
        labels = labels.to(device)
        output = model(padded_text, attention_masks)
        loss = criterion(output, labels)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        if max_norm:
            clip_grad_norm_(model.parameters(), max_norm=max_norm)

    return total_loss / len(loader)

def gradient_l1unstructured(model, module, name, amount, loader, criterion, optimizer, device,
                            epoch=1, max_norm=None):
    # clean_data = read_data(data_path)
    # train_loader_clean = packDataset_util.get_loader(clean_data, shuffle=True, batch_size=BATCH_SIZE)
    param = getattr(module, name)
    grad_acc = torch.zeros_like(param)
    for i in range(epoch):
        avg_loss = train(model, loader, criterion, optimizer, device, max_norm)
        # param = getattr(module, name)
        grad_acc += param.grad.detach()
    grad_acc /= epoch
    L1Unstructured.apply(
        module, name, amount=amount, importance_scores=1 / (grad_acc.abs()+1e-5)
    )

    return model


# class ActivationStructured(BasePruningMethod):
#     PRUNING_TYPE = "structured"
#
#     def __init__(self, amount, activations, dim=-1):
#         # Check range of validity of amount
#         super().__init__()
#         self.amount = amount
#         self.activations = activations
#         self.dim = dim
#
#     def compute_mask(self, t, default_mask):
#         # Check that the amount of channels to prune is not > than the number of
#         # channels in t along the dim to prune
#         tensor_size = t.shape[self.dim]
#
#         assert tensor_size == len(self.activations)
#
#         # Structured pruning prunes entire channels so we need to know the
#         # L_n norm along each channel to then find the topk based on this
#         # metric
#         norm = _compute_norm(t, self.n, self.dim)
#         # largest=True --> top k; largest=False --> bottom k
#         # Keep the largest k channels along dim=self.dim
#         topk = torch.topk(norm, k=nparams_tokeep, largest=True)
#
#         # topk will have .indices and .values
#
#         # Compute binary mask by initializing it to all 0s and then filling in
#         # 1s wherever topk.indices indicates, along self.dim.
#         # mask has the same shape as tensor t
#         def make_mask(t, dim, indices):
#             # init mask to 0
#             mask = torch.zeros_like(t)
#             # e.g.: slc = [None, None, None], if len(t.shape) = 3
#             slc = [slice(None)] * len(t.shape)
#             # replace a None at position=dim with indices
#             # e.g.: slc = [None, None, [0, 2, 3]] if dim=2 & indices=[0,2,3]
#             slc[dim] = indices
#             # use slc to slice mask and replace all its entries with 1s
#             # e.g.: mask[:, :, [0, 2, 3]] = 1
#             mask[slc] = 1
#             return mask
#
#         if nparams_toprune == 0:  # k=0 not supported by torch.kthvalue
#             mask = default_mask
#         else:
#             mask = make_mask(t, self.dim, topk.indices)
#             mask *= default_mask.to(dtype=mask.dtype)
#
#         return mask
#
#     [docs] @ classmethod
#
#     def apply(cls, module, name, amount, n, dim, importance_scores=None):
#         return super(LnStructured, cls).apply(
#             module,
#             name,
#             amount=amount,
#             n=n,
#             dim=dim,
#             importance_scores=importance_scores,
#         )



