"""
Convert a DualWeight model to codebook lazy mode for inference memory saving.
"""
import torch
from dual_weight_layers import DualWeightWeaveFormer, DualWeightLinear
from codebook_lazy_dual import CodebookDualWeightLinear, make_codebook_dual_weight_layer
from dual_weight_layers import DualWeightTransformerEncoderLayer, DualWeightMultiheadAttention

def convert_model_to_codebook(model, codebook_capacity=1024):
    """
    Walk model and replace DualWeightLinear with CodebookDualWeightLinear,
    compressing weights into layered codebooks.
    Returns new model with same architecture but lazy dequantization.
    """
    # For simplicity, we rebuild model by cloning structure
    # Here we do in-place replacement via recursion
    def _replace(module):
        for name, child in module.named_children():
            if isinstance(child, DualWeightLinear):
                # Create codebook version
                new_lin = CodebookDualWeightLinear(
                    in_features=child.in_features,
                    out_features=child.out_features,
                    bias=child.master_bias is not None,
                    delta_decay=child.delta_decay,
                    delta_clip=child.delta_clip,
                    use_codebook=True,
                    codebook_capacity=codebook_capacity,
                )
                # Copy biases
                if child.master_bias is not None:
                    new_lin.master_bias.data.copy_(child.master_bias.data)
                    new_lin.delta_bias.data.copy_(child.delta_bias.data)
                new_lin.convert_from_weights(child.master_weight, child.delta_weight)
                setattr(module, name, new_lin)
            else:
                _replace(child)
    _replace(model)
    return model

if __name__ == "__main__":
    model = DualWeightWeaveFormer(dim=64, num_layers=2, vocab_size=128, use_dual_weights=True)
    model.eval()
    x = torch.randint(0,128,(1,16))
    with torch.no_grad():
        out_before = model(x)
    print("Before conversion, output mean:", out_before.mean().item())
    model = convert_model_to_codebook(model, codebook_capacity=512)
    model.eval()
    with torch.no_grad():
        out_after = model(x)
    print("After conversion, output mean:", out_after.mean().item())
    print("Max diff:", (out_before - out_after).abs().max().item())
