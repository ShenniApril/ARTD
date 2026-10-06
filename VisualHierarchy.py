# make use of ViT with different layers
import torch
import torch.nn as nn
import torch.nn.functional as F

class ViT(nn.Module):
    def __init__(
        self,
        vit, # pretrained ViT model
        layer_ids=(3, 7, 11),
        vit_dim=768,
        align_dim=512,
    ):
        super().__init__()

        self.vit = vit
        self.layer_ids = tuple(layer_ids)

        for param in self.vit.parameters():
            param.requires_grad = False

        self.vit.eval()
        
    def train(self, mode=True):
        super().train(mode)
        self.vit.eval()
        return self
    
    @torch.no_grad()
    def extract_features(self, image):
        x = self.vit.patch_embed(image)
        x = self.vit._pos_embed(x)
        x = self.vit.patch_drop(x)
        x = self.vit.norm_pre(x)

        outputs = []

        for index, block in enumerate(self.vit.blocks):
            x = block(x)

            if index in self.layer_ids:
                outputs.append(x)

        return outputs
    
    def pool_tokens(self, tokens):
        if self.has_cls_token:
            patch_tokens = tokens[:, 1:]
        else:
            patch_tokens = tokens

        return patch_tokens.mean(dim=1)
    
    self.projectors = nn.ModuleList([
        nn.Sequential(
            nn.LayerNorm(vit_dim),
            nn.Linear(vit_dim, align_dim),
        )
        for _ in layer_ids
    ])
    
    @torch.no_grad()
    def extract_frozen_features(self, image):
        return self.extract_features(image)

    def forward(self, image):
        raw_features = self.extract_frozen_features(image)

        targets = []

        for feature, projector in zip(
            raw_features,
            self.projectors,
        ):
            pooled = self.pool_tokens(feature)
            target = projector(pooled)
            target = F.normalize(target, dim=-1)
            targets.append(target)

        return {
            "aligned": torch.stack(targets, dim=1),
            "raw_features": raw_features,
        }