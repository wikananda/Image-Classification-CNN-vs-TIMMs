from models.heads import MLP
import torch
import timm

class TimmClassifier(torch.nn.Module):
    def __init__(
        self,
        name: str = "resnet50",
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        cache_dir: str = 'checkpoints',
        **create_kwargs, # call drop_rate, global_pool,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            name,
            pretrained=pretrained,
            num_classes=0, # drop the classifier
            global_pool="",
            cache_dir=cache_dir,
            **create_kwargs,
        )
        if freeze_backbone:
            self._freeze(self.backbone)

        self.feature_dim = self.backbone.num_features
        self.head = MLP(in_features=self.feature_dim ,num_classes=num_classes)

    def _freeze(self, module):
       for p in module.parameters():
           p.requires_grad = False

    def forward(self, x):
        feats = self.backbone.forward_features(x)
        if feats.dim() == 2:
            feats = feats.unsqueeze(-1).unsqueeze(-1)
        return self.head(feats)
    
