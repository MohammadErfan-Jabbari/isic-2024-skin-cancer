import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# ===========================
# 1. GeM Pooling (Generalized Mean Pooling)
# ===========================
class GeM(nn.Module):
    """
    Generalized Mean Pooling.
    p=1: Average Pooling
    p=inf: Max Pooling
    p is learnable.
    """
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
        
    def __repr__(self):
        return self.__class__.__name__ + \
                '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + \
                ', ' + 'eps=' + str(self.eps) + ')'

# ===========================
# 2. SE Block (Squeeze-and-Excitation)
# ===========================
class SEBlock(nn.Module):
    """
    Re-calibrates feature importance after concatenation.
    """
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [Batch, Channel]
        b, c = x.size()
        y = x.view(b, c, 1)
        y = self.fc(y).view(b, c)
        return x * y

# ===========================
# 3. Advanced Hybrid Model
# ===========================
class AdvancedHybridModel(nn.Module):
    def __init__(self, model_name, metadata_dim, pretrained=True, image_size=None):
        super().__init__()
        
        # --- Backbone ---
        # We remove the classifier and global pool to add our own GeM
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=0,
            global_pool='' # Disable default pooling
        )
        
        # Determine feature dimension
        with torch.no_grad():
            # Dummy pass to get feature map size
            dummy_size = image_size if image_size else 224
            dummy = torch.randn(2, 3, dummy_size, dummy_size)
            feats = self.backbone(dummy)
            # feats shape: [B, C, H, W]
            self.img_feature_dim = feats.shape[1]
            
        # --- GeM Pooling ---
        self.pool = GeM()
        self.img_bn = nn.BatchNorm1d(self.img_feature_dim)
        
        # --- Metadata Branch ---
        self.meta_net = nn.Sequential(
            nn.Linear(metadata_dim, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(), # Swish activation (modern standard)
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.2)
        )
        
        # --- Fusion & Classifier ---
        combined_dim = self.img_feature_dim + 128
        
        self.fusion_se = SEBlock(combined_dim, reduction=8)
        
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
        
    def forward(self, img, meta):
        # 1. Image Features
        x_img = self.backbone(img)      # [B, C, H, W]
        x_img = self.pool(x_img)        # [B, C, 1, 1]
        x_img = x_img.flatten(1)        # [B, C]
        x_img = self.img_bn(x_img)
        
        # 2. Metadata Features
        x_meta = self.meta_net(meta)    # [B, 128]
        
        # 3. Concatenation
        x_combined = torch.cat([x_img, x_meta], dim=1)
        
        # 4. Attention (SE Block)
        # Allows model to weigh "Image says X" vs "Meta says Y"
        x_combined = self.fusion_se(x_combined)
        
        # 5. Classification
        output = self.classifier(x_combined)
        return output
