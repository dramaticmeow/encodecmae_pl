from .transformers import TransformerEncoder, SinusoidalPositionalEmbeddings, MultiHeadAttention

from .encoders import WavEncoder, EncodecEncoder
from .targets import EncodecQuantizer
from .masks import TimeGapMask, PatchoutMask
from .heads import FrameLevelClassificationHead
from .losses import EnCodecMAEClassificationLoss
from .mae import EncodecMAE
from functools import partial

import torch
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
    sin_pos = partial(SinusoidalPositionalEmbeddings, embedding_dim=args.transformer.dim)

    if args.input == 'encodec':
        wav_encoder = partial(WavEncoder, encoder=EncodecEncoder, pre_net=None, post_net=partial(torch.nn.Linear, in_features=args.wav_encoder.wav_feature_dim, out_features=args.transformer.dim))
    elif args.input == 'mel':
        wav_encoder = partial(WavEncoder, encoder=torch.nn.Identity, pre_net=None, post_net=partial(torch.nn.Linear, in_features=args.mel.num_bins, out_features=args.transformer.dim), fs=args.wav_encoder.fs, hop_length=args.wav_encoder.hop_length, key_in = 'wav_features', key_out = 'wav_features')
    patch_out_masker = partial(PatchoutMask, masker=partial(TimeGapMask, p_mask=args.masking.prop, gap_size=args.masking.gap_size), positional_encoder=sin_pos)

    trans_enc = partial(TransformerEncoder, model_dim=args.transformer.dim, num_layers=args.transformer.enc_num_layers, attention_layer=attn_enc, compile=False, key_in='visible_tokens', key_padding_mask='visible_padding_mask', key_out='decoder_in', key_transformer_in=None, key_transformer_out='visible_embeddings', post_net=partial(torch.nn.Linear,in_features=args.transformer.dim, out_features=args.transformer.dim))

    trans_dec = partial(TransformerEncoder, model_dim=args.transformer.dim, num_layers=args.transformer.dec_num_layers, attention_layer=attn_dec, compile=False, key_in='decoder_in', key_padding_mask='feature_padding_mask', key_out='decoder_out', positional_encoder=sin_pos)

    if args.input == 'encodec':
        quantizer = partial(EncodecQuantizer, n=args.quantizer.num_encodec_targets, key_in = 'wav_features_encoder_out', return_only_last = False)
    elif args.input == 'mel':
        quantizer = partial(EncodecQuantizer, n=args.quantizer.num_encodec_targets, key_in = 'wav', use_encodec_encoder=True)

    ret = EncodecMAE(
        wav_encoder=wav_encoder,
        target_encoder=quantizer,
        masker=patch_out_masker,
        visible_encoder=trans_enc,
        decoder=trans_dec,
        head=partial(FrameLevelClassificationHead, model_dim=args.transformer.dim, num_tokens=args.target.num_tokens, num_streams=args.target.num_heads),
        loss=partial(EnCodecMAEClassificationLoss, masked_weight=args.loss.masked_weight, quantizer_weights=args.loss.quantizer_weights),
        optimizer=partial(AdamW, lr=args.optim.max_lr, betas=(args.optim.beta1, args.optim.beta2), weight_decay=args.optim.weight_decay),
    )
    return ret

if __name__ == '__main__':
    encodec_mae = build_EncodecMAE(OmegaConf.load('/2214/dongyuanliang/encodecmae_pl/config/encodecmae_large.yaml'))
    print(encodec_mae)