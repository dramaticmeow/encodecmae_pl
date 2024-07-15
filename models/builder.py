from .transformers import TransformerEncoder, SinusoidalPositionalEmbeddings, MultiHeadAttention

from .encodecmae.encoders import EncodecEncoder
from .encodecmae.targets import EncodecQuantizer
from .encodecmae.masking import TimeGapMask
from .encodecmae.heads import FrameLevelClassificationHead
from .encodecmae.mae import EncodecMAE
from functools import partial

from torch.optim import AdamW
from omegaconf import OmegaConf
def build_EncodecMAE(args):
    # trans_enc = TransformerEncoder(
    #     model_dim=args.transformer.dim, 
    #     num_layers=args.transformer.enc_num_layers, 
    #     attention_layer=MultiHeadAttention(model_dim=args.transformer.dim, num_heads=args.transformer.enc_num_heads)
    #     )
    # trans_dec = TransformerEncoder(
    #     model_dim=args.transformer.dim, 
    #     num_layers=args.transformer.dec_num_layers, 
    #     attention_layer=MultiHeadAttention(model_dim=args.transformer.dim, num_heads=args.transformer.dec_num_heads)
    #     )
    # convert to partial
    attn_enc = partial(MultiHeadAttention, model_dim=args.transformer.dim, num_heads=args.transformer.enc_num_heads)
    attn_dec = partial(MultiHeadAttention, model_dim=args.transformer.dim, num_heads=args.transformer.dec_num_heads)
    trans_enc = partial(TransformerEncoder, model_dim=args.transformer.dim, num_layers=args.transformer.enc_num_layers, attention_layer=attn_enc)
    trans_dec = partial(TransformerEncoder, model_dim=args.transformer.dim, num_layers=args.transformer.dec_num_layers, attention_layer=attn_dec)

    ret = EncodecMAE(
        wav_encoder=EncodecEncoder,
        target_encoder=partial(EncodecQuantizer, n=args.quantizer.num_encodec_targets),
        visible_encoder=trans_enc,
        decoder=trans_dec,
        positional_encoder=partial(SinusoidalPositionalEmbeddings, embedding_dim=args.transformer.dim),
        masker=partial(TimeGapMask, gap_size=args.masking.gap_size, mask_prop=args.masking.prop),
        head=partial(FrameLevelClassificationHead, model_dim=args.transformer.dim, num_tokens=args.target.num_tokens, num_streams=args.target.num_heads),
        optimizer=partial(AdamW, lr=args.optim.max_lr, betas=(args.optim.beta1, args.optim.beta2), weight_decay=args.optim.weight_decay),
        masked_weight = args.loss.masked_weight,
        quantizer_weights = args.loss.quantizer_weights,
    )
    return ret

if __name__ == '__main__':
    encodec_mae = build_EncodecMAE(OmegaConf.load('/data41/private/dongyuanliang/encodecmae/encodecmae_pl/config/encodecmae_base.yaml'))
    print(encodec_mae)