# MONO_DATASET='en:./newdata/OPUS-100/opus.en-hi-train.en.filtered.pth,,;hi:./newdata/OPUS-100/opus.en-hi-train.hi.filtered.pth,,'
# PARA_DATASET='en-hi:./newdata/OPUS-100/opus.en-hi-train.XX.filtered.pth,./newdata/OPUS-100/opus.en-hi-valid.XX.filtered.pth,./newdata/OPUS-100/opus.en-hi-test.XX.filtered.pth'
# PRETRAINED='./newdata/all.256.vec'

MONO_DATASET='en:./newdata/combinedcs/combinedcs.en.train.pth,,;hi:./newdata/combinedcs/combinedcs.hi.train.pth,,'
PARA_DATASET='en-hi:./newdata/combinedcs/combinedcs.XX.train.pth,./newdata/combinedcs/combinedcs.XX.valid.pth,./newdata/combinedcs/combinedcs.XX.test.pth'
PRETRAINED='./newdata/all.256.vec'

CUDA_VISIBLE_DEVICES=1 python3 main.py \
--exp_name temp \
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
--epoch_size 10000 \
--stopping_criterion bleu_en_hi_valid,50 \
--freeze_enc_emb False \
--freeze_dec_emb False \
--reload_model newdumped/B_combinedcs/best-bleu_en_hi_valid.pth \
--reload_enc True \
--reload_dec True \
--eval_only True
