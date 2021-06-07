PARA_DATASET='en-hi:./data/IITB/IITB.XX.train.tok.pth,./data/IITB/IITB.XX.valid.tok.pth,./data/IITB/IITB.XX.test.tok.pth'
PRETRAINED='./data/all.256.vec'

CUDA_VISIBLE_DEVICES=1 python3 main.py \
--exp_name iitb_ce \
--transformer True \
--n_enc_layers 3 \
--n_dec_layers 3 \
--share_enc 2 \
--share_dec 2 \
--share_lang_emb True \
--share_output_emb True \
--emb_dim 256 \
--langs 'en,hi' \
--n_para -1 \
--para_dataset $PARA_DATASET \
--para_directions 'en-hi,hi-en' \
--pretrained_emb $PRETRAINED \
--pretrained_out True \
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
--reload_model dumped/iitb_ce/qgxejk4wsj/best-bleu_en_hi_valid.pth \
--reload_enc True \
--reload_dec True