# import torch
# import numpy as np
#
# from sklearn.metrics import log_loss
#
#
# def check_accuracy(
#         loader, model, loss_fn, input_shape=None, toggle_eval=True, print_accuracy=True
# ):
#     """
#     Check accuracy of model on data from loader
#     """
#     if toggle_eval:
#         model.eval()
#     device = next(model.parameters()).device
#     num_correct = 0
#     num_samples = 0
#
#     y_preds = []
#     y_true = []
#
#     with torch.no_grad():
#         for x, y in loader:
#             x = x.to(device=device)
#             y = y.to(device=device)
#             if input_shape:
#                 x = x.reshape(x.shape[0], *input_shape)
#             scores = model(x)
#             scores = torch.argmax(scores, dim=1).float()
#             # predictions = torch.sigmoid(scores) > 0.5
#             # y_preds.append(torch.clip(torch.sigmoid(scores), 0.005, 0.995).cpu().numpy())
#             y_true.append(y.cpu().numpy())
#             num_correct += (scores == y).sum()
#             num_samples += scores.size(0)
#
#     accuracy = num_correct / num_samples
#
#     if toggle_eval:
#         model.train()
#
#     if print_accuracy:
#         print(f"Accuracy: {accuracy * 100:.2f}%")
#         # print(log_loss(np.concatenate(y_true, axis=0), np.concatenate(y_preds, axis=0)))
#
#     return accuracy
# 
#
# def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
#     print("=> Saving checkpoint")
#     torch.save(state, filename)
#
#
# def load_checkpoint(checkpoint, model):
#     print("=> Loading checkpoint")
#     model.load_state_dict(checkpoint["state_dict"])
