MONO_DATASET='en:./common/PRETRAIN_COMBINED/en.train.pth,,;hi:./common/PRETRAIN_COMBINED/hi.train.pth,,'
PARA_DATASET='en-hi:./common/PRETRAIN_COMBINED/XX.train.pth,./common/PRETRAIN_COMBINED/XX.valid.pth,./common/PRETRAIN_COMBINED/XX.test.pth'
PRETRAINED='./common/all.256.vec'

CUDA_VISIBLE_DEVICES=3 python3 main.py \
--exp_name pretrain_combined \
--transformer True \
--n_enc_layers 3 \
--n_dec_layers 3 \
--share_enc 2 \
--share_dec 2 \
--share_lang_emb True \
--share_output_emb True \
--emb_dim 256 \
--langs 'en,hi' \
--n_mono -1 \
--n_para -1 \
--mono_dataset $MONO_DATASET \
--para_dataset $PARA_DATASET \
--mono_directions 'en,hi' \
--para_directions 'en-hi,hi-en' \
--word_shuffle 3 \
--word_dropout 0.1 \
--word_blank 0.2 \
--pretrained_emb $PRETRAINED \
--pretrained_out True \
--lambda_xe_mono 1 \
--lambda_xe_para 1 \
--otf_num_processes 30 \
--otf_sync_params_every 1000 \
--enc_optimizer adam,lr=0.0001 \
--group_by_size True \
--batch_size 16 \
--epoch_size 100000 \
--stopping_criterion bleu_en_hi_valid,5 \
--freeze_enc_emb False \
--freeze_dec_emb False
# --eval_only True \
# --reload_model dumped/opus_pretrain_temp/best-bleu_en_hi_valid.pth \
# --reload_enc True \
# --reload_dec True