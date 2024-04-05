import torch
import timm
import numpy as np
import math

from einops import repeat, rearrange
from einops.layers.torch import Rearrange

from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block

def random_indexes(size : int):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes

def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))

class PatchShuffle(torch.nn.Module):
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio

    def forward(self, patches : torch.Tensor):
        T, B, C = patches.shape
        remain_T = int(T * (1 - self.ratio))

        indexes = [random_indexes(T) for _ in range(B)]
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)

        patches = take_indexes(patches, forward_indexes)
        patches = patches[:remain_T]

        return patches, forward_indexes, backward_indexes

class MAE_Encoder(torch.nn.Module):
    def __init__(self,
                 image_size=512, # H(W)
                 patch_size=32, # H(W)/16
                 emb_dim=768, # 192 768 1024 1280
                 num_layer=12, # 12 12 24 32
                 num_head=12, #3 12 16 16
                 mask_ratio=0.75,
                 ) -> None:
        super().__init__()

        self.emb_dim = emb_dim
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2, 1, emb_dim))
        self.shuffle = PatchShuffle(mask_ratio)

        self.patchify = torch.nn.Conv2d(3, emb_dim, patch_size, patch_size)

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.layer_norm = torch.nn.LayerNorm(emb_dim)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, img):
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding

        patches, forward_indexes, backward_indexes = self.shuffle(patches)

        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')

        return features, backward_indexes

class MAE_Decoder(torch.nn.Module):
    def __init__(self,
                 image_size=512, # H(W)
                 patch_size=32, # H(W)/16
                 emb_dim=512, # 192 512 512 512
                 num_layer=8, # 4 8 8 8
                 num_head=16, # 3 16 16 16
                 ) -> None:
        super().__init__()

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2 + 1, 1, emb_dim))

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.head = torch.nn.Linear(emb_dim, 3 * patch_size ** 2)
        self.patch2img = Rearrange('(h w) b (c p1 p2) -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=image_size//patch_size)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features, backward_indexes):
        T = features.shape[0]
        backward_indexes = torch.cat([torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes), backward_indexes + 1], dim=0)
        features = torch.cat([features, self.mask_token.expand(backward_indexes.shape[0] - features.shape[0], features.shape[1], -1)], dim=0)
        features = take_indexes(features, backward_indexes)
        features = features + self.pos_embedding

        features = rearrange(features, 't b c -> b t c')
        features = self.transformer(features)
        features = rearrange(features, 'b t c -> t b c')
        features = features[1:] # remove global feature

        patches = self.head(features)
        mask = torch.zeros_like(patches)
        mask[T-1:] = 1
        mask = take_indexes(mask, backward_indexes[1:] - 1)
        img = self.patch2img(patches)
        mask = self.patch2img(mask)

        return img, mask
    
class MAE_ViT(torch.nn.Module):
    def __init__(self,
                 image_size=512, # H(W)
                 patch_size=32, # H(W)/16
                 emb_dim=768, # 192 768 1024 1280
                 encoder_layer=12, # 12 12 24 32
                 encoder_head=12, # 3 12 16 16
                 decoder_layer=8, # 4 8 8 8
                 decoder_head=16, # 3 16 16 16
                 mask_ratio=0.75,
                 ) -> None:
        super().__init__()

        self.encoder = MAE_Encoder(image_size, patch_size, emb_dim, encoder_layer, encoder_head, mask_ratio)
        self.decoder = MAE_Decoder(image_size, patch_size, emb_dim, decoder_layer, decoder_head)

    def forward(self, img):
        features, backward_indexes = self.encoder(img)
        predicted_img, mask = self.decoder(features,  backward_indexes)
        return predicted_img, mask

class ViT_Classifier(torch.nn.Module):
    def __init__(self, encoder : MAE_Encoder, num_classes=10) -> None:
        super().__init__()
        self.cls_token = encoder.cls_token
        self.pos_embedding = encoder.pos_embedding
        self.patchify = encoder.patchify
        self.transformer = encoder.transformer
        self.layer_norm = encoder.layer_norm
        self.head = torch.nn.Linear(self.pos_embedding.shape[-1], num_classes)

    def forward(self, img):
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding

        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')
        logits = self.head(features[0])
        return logits

class ViT_Segmentor(torch.nn.Module):
    def __init__(self, encoder : MAE_Encoder, out_channels=10, downsample_factor=16, features=[512, 256, 128, 64, 32]) -> None: 
        super().__init__()
        self.cls_token = encoder.cls_token
        self.pos_embedding = encoder.pos_embedding
        self.patchify = encoder.patchify
        self.transformer = encoder.transformer
        self.layer_norm = encoder.layer_norm
        self.downsample_factor = downsample_factor
        in_channels = encoder.emb_dim
        self.alpha = torch.nn.Parameter(torch.tensor(0.0))
        self.beta = torch.nn.Parameter(torch.tensor(0.0))
            
        self.decoder_1 = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels, features[0], 3, padding=1),
                    torch.nn.BatchNorm2d(features[0]),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
                )
        self.decoder_2 = torch.nn.Sequential(
                    torch.nn.Conv2d(features[0], features[1], 3, padding=1),
                    torch.nn.BatchNorm2d(features[1]),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
                )
        self.decoder_3 = torch.nn.Sequential(
                    torch.nn.Conv2d(features[1], features[2], 3, padding=1),
                    torch.nn.BatchNorm2d(features[2]),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
                )
        self.decoder_4 = torch.nn.Sequential(
                    torch.nn.Conv2d(features[2], features[3], 3, padding=1),
                    torch.nn.BatchNorm2d(features[3]),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
                )

        self.decoder_5 = torch.nn.Sequential(
                    torch.nn.Conv2d(features[3], features[4], 3, padding=1),
                    torch.nn.BatchNorm2d(features[4]),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
                )
        
        self.scene_1 = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels, 1, 1, padding=0),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
                )
        
        self.scene_2 = torch.nn.Sequential(
                    torch.nn.Conv2d(features[0], 1, 1, padding=0),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
                )
        
        self.scene_3 = torch.nn.Sequential(
                    torch.nn.Conv2d(features[1], 1, 1, padding=0),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
                )
        
        self.scene_4 = torch.nn.Sequential(
                    torch.nn.Conv2d(features[2], 1, 1, padding=0),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
                )
        
        self.scene_5 = torch.nn.Sequential(
                    torch.nn.Conv2d(features[3], 1, 1, padding=0),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
                )

        self.origin_scene_embedding = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels, out_channels, 1, padding=0),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Upsample(scale_factor=self.downsample_factor*2, mode="bilinear", align_corners=True)
                )
    

        if self.downsample_factor == 8:
            self.final_out = torch.nn.Conv2d(features[-2], out_channels, 3, padding=1)
        elif self.downsample_factor == 16:
            self.final_out = torch.nn.Conv2d(features[-1], out_channels, 3, padding=1)
        else:
            raise ValueError("downsample factor which depends on your image size and patch size can only be 8 or 16.")

    def forward(self, img):
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding

        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> b c t')
        features = rearrange(features, 'b c (h w) -> b c h w', h=int(math.sqrt(features.shape[-1])), w=int(math.sqrt(features.shape[-1])))

        # Scene Based Decoder
        global_scene_embedding = self.origin_scene_embedding(features)

        feature_scene1 = self.scene_1(features)
        x = self.decoder_1(features)

        feature_scene2 = self.scene_2(x+feature_scene1)
        x = self.decoder_2(x+feature_scene1)

        feature_scene3 = self.scene_3(x+feature_scene2)
        x = self.decoder_3(x+feature_scene2)

        feature_scene4 = self.scene_4(x+feature_scene3)
        x = self.decoder_4(x+feature_scene3)

        if self.downsample_factor == 16:
            feature_scene5 = self.scene_5(x+feature_scene4)
            x = self.decoder_5(x+feature_scene4)
            x = self.final_out(x+feature_scene5)
        else:
            x = self.final_out(x+feature_scene4)

        final_out = self.alpha * x + self.beta * global_scene_embedding

        return final_out, x, global_scene_embedding
    
if __name__ == '__main__':
    model_type = "Segmentation"
    if model_type == "Pretrained":
        print("Pretrained")
        shuffle = PatchShuffle(0.75)
        a = torch.rand(16, 2, 10)
        b, forward_indexes, backward_indexes = shuffle(a)
        print(b.shape)

        img = torch.rand(2, 3, 32, 32)
        encoder = MAE_Encoder()
        decoder = MAE_Decoder()
        features, backward_indexes = encoder(img)
        print(forward_indexes.shape)
        predicted_img, mask = decoder(features, backward_indexes)
        print(predicted_img.shape)
        loss = torch.mean((predicted_img - img) ** 2 * mask / 0.75)
        print(loss)
    
    elif model_type == "Classify":
        print("Classify")
        img = torch.rand(2, 3, 512, 512)
        encoder = MAE_Encoder()
        model = ViT_Classifier(encoder, num_classes=10)
        output = model(img)
        print(output.shape)
    
    elif model_type == "Segmentation":
        print("Segmentation")
        img = torch.rand(2, 3, 512, 512)
        encoder = MAE_Encoder(patch_size=32)
        model = ViT_Segmentor(encoder, out_channels=10, downsample_factor=16)
        output = model(img)
        print(output.shape)

    else:
        raise ValueError("model_type can only be Pretrained, Classify or Segmentation.")

# def mae_vit_base_patch16_dec512d8b(**kwargs):
#     model = MaskedAutoencoderViT(
#         patch_size=16, embed_dim=768, depth=12, num_heads=12,
#         decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
#         mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model


# def mae_vit_large_patch16_dec512d8b(**kwargs):
#     model = MaskedAutoencoderViT(
#         patch_size=16, embed_dim=1024, depth=24, num_heads=16,
#         decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
#         mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model


# def mae_vit_huge_patch14_dec512d8b(**kwargs):
#     model = MaskedAutoencoderViT(
#         patch_size=14, embed_dim=1280, depth=32, num_heads=16,
#         decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
#         mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model
    