import torch
import torch.nn as nn
import torch.nn.functional as F

class AlignmentModel(nn.Module):
    def __init__(self, eeg_encoder, visual_teacher):
        super().__init__()
        self.eeg_encoder = eeg_encoder
        self.visual_teacher = visual_teacher

    def forward(self, eeg, image, input_chans=None):
        eeg_out = self.eeg_encoder(eeg, input_chans=input_chans)
        visual_out = self.visual_teacher(image)
        loss_hier = (1.0 - F.cosine_similarity(eeg_out["aligned"], visual_out["aligned"],dim=-1)).mean()

        return {
            "eeg": eeg_out,
            "visual": visual_out,
            "hierarchy_loss": loss_hier,
        }