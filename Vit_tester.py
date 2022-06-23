import timm
import torch

model = timm.create_model('vit_base_patch16_224', pretrained=True).eval()
x = torch.randn(1,3,224,224)
out = model(x)

# print(out)
# print(out.shape)
print(model)