MONO_DATASET='en:./data/IITB/mono/en.tok.pth,,;hi:./data/IITB/mono/hi.tok.pth,,'
PARA_DATASET='en-hi:,./data/IITB/para/XX.valid.tok.pth,./data/IITB/para/XX.test.tok.pth'
PRETRAINED='./data/IITB/mono/all.vec'

CUDA_VISIBLE_DEVICES=3 python3 main.py \
--exp_name IITB-Parallel \
--transformer True \
--n_enc_layers 3 \
--n_dec_layers 3 \
--share_enc 2 \
--share_dec 2 \
--share_lang_emb True \
--share_output_emb True \
--langs 'en,hi' \
--n_mono -1 \
--mono_dataset $MONO_DATASET \
--para_dataset $PARA_DATASET \
--mono_directions 'en,hi' \
--word_shuffle 3 \
--word_dropout 0.1 \
--word_blank 0.2 \
--pivo_directions 'en-hi-en,hi-en-hi' \
--pretrained_emb $PRETRAINED \
--pretrained_out True \
--lambda_xe_mono '0:1,10000:0.1,30000:0' \
--lambda_xe_otfd 1 \
--otf_num_processes 30 \
--otf_sync_params_every 1000 \
--enc_optimizer adam,lr=0.0001 \
--group_by_size True \
--batch_size 8 \
--epoch_size 50000 \
--stopping_criterion bleu_en_hi_valid,10 \
--freeze_enc_emb False \
--freeze_dec_emb False