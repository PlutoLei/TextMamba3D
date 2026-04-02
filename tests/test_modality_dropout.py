import torch
from models.textmamba3d import TextMamba3D

def test_vision_dropout_attribute():
    model = TextMamba3D(img_size=(32,32,32), embed_dim=48, depths=[2,2,2,2],
                        text_embed_dim=256, use_pretrained_text=False)
    assert hasattr(model, 'vision_dropout_rate')
    assert model.vision_dropout_rate == 0.0

def test_vision_dropout_zeros_features():
    torch.manual_seed(42)
    model = TextMamba3D(img_size=(32,32,32), embed_dim=48, depths=[2,2,2,2],
                        text_embed_dim=256, use_pretrained_text=False)
    model.train()
    model.vision_dropout_rate = 1.0  # 100% dropout
    img = torch.randn(1, 4, 32, 32, 32)
    text_ids = torch.randint(0, 1000, (1, 16))
    mask = torch.ones(1, 16)
    # Should not crash even with all-zero vision features
    out = model(img, text_ids, mask, use_text=True)
    assert out.shape[0] == 1  # batch dim preserved
    assert not torch.isnan(out).any(), "Output should not contain NaN"

def test_no_dropout_in_eval():
    torch.manual_seed(42)
    model = TextMamba3D(img_size=(32,32,32), embed_dim=48, depths=[2,2,2,2],
                        text_embed_dim=256, use_pretrained_text=False)
    model.eval()
    model.vision_dropout_rate = 1.0  # even at 100%, eval should not drop
    img = torch.randn(1, 4, 32, 32, 32)

    out1 = model(img, use_text=False)
    out2 = model(img, use_text=False)
    # In eval mode, same input should give same output (no random dropout)
    assert torch.allclose(out1, out2, atol=1e-5)
