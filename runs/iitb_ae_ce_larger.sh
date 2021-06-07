MONO_DATASET='en:./data/IITB/IITB.en.train.tok.pth,,;hi:./data/IITB/IITB.hi.train.tok.pth,,'
PARA_DATASET='en-hi:./data/IITB/IITB.XX.train.tok.pth,./data/IITB/IITB.XX.valid.tok.pth,./data/IITB/IITB.XX.test.tok.pth'
PRETRAINED='./data/all.512.vec'

CUDA_VISIBLE_DEVICES=2 python3 main.py \
--exp_name iitb_ae_ce_larger \
--transformer True \
--n_enc_layers 4 \
--n_dec_layers 4 \
--share_enc 3 \
--share_dec 3 \
--share_lang_emb True \
--share_output_emb True \
--emb_dim 512 \
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
--batch_size 32 \
--epoch_size 50000 \
--stopping_criterion bleu_en_hi_valid,10 \
--freeze_enc_emb False \
--freeze_dec_emb False \
--eval_only True \
--reload_model dumped/iitb_ae_ce_larger/sye2tj41r4/best-bleu_en_hi_valid.pth \
--reload_enc True \
--reload_dec True