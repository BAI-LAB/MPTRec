import torch
from sklearn.metrics import roc_auc_score


class EvaluationManager:
    def __init__(self, model, dataloader):
        self.model = model
        self.dataloader = dataloader

    @torch.no_grad()
    def evaluation(self):
        self.model.eval()
        device = next(self.model.parameters()).device
        auc_score_1, auc_score_2 = 0, 0
        for y_1, y_2, features in self.dataloader:
            for key in features.keys():
                features[key] = features[key].to(device)
            pred_1, pred_2 = self.model(features)

            try:
                auc_score_1 += roc_auc_score(y_1.int(), pred_1.cpu())
            except ValueError:
                auc_score_1 += 1
            try:
                auc_score_2 += roc_auc_score(y_2.int(), pred_2.cpu())
            except ValueError:
                auc_score_2 += 1

        return auc_score_1 / len(self.dataloader), auc_score_2 / len(self.dataloader)