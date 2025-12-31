# 对于opt模型，探索4bit量化的情况下，各种数据格式，ppl实验
#region
# 12.1 FP16
CUDA_VISIBLE_DEVICES=4 python Iron_weight_only_quant/main.py --model_path /home/data/meta-llama/opt/6.7b/ --w_bits 16 --eval_mode ppl --datasets wikitext ptb c4 --output_file cuda4.json
# 12.2 INT4
CUDA_VISIBLE_DEVICES=4 python Iron_weight_only_quant/main.py --model_path /home/data/meta-llama/opt/6.7b/ --w_bits 4 --w_group_size 128 --w_symmetric --eval_mode ppl --datasets wikitext ptb c4 --output_file cuda4.json
# 12.3 BFP4
CUDA_VISIBLE_DEVICES=4 python Iron_weight_only_quant/main.py --model_path /home/data/meta-llama/opt/6.7b/ --w_format bfp --w_bits 4 --w_group_size 128 --w_symmetric --eval_mode ppl --datasets wikitext ptb c4 --output_file cuda4.json
# 12.4 FP4E1M2
CUDA_VISIBLE_DEVICES=4 python Iron_weight_only_quant/main.py --model_path /home/data/meta-llama/opt/6.7b/ --w_format fp4 --w_bits 4 --fp4_exp_bits 1 --fp4_mantissa_bits 2 --w_group_size 128 --w_symmetric --eval_mode ppl --datasets wikitext ptb c4 --output_file cuda4.json
# 12.5 FP4E2M1
CUDA_VISIBLE_DEVICES=4 python Iron_weight_only_quant/main.py --model_path /home/data/meta-llama/opt/6.7b/ --w_format fp4 --w_bits 4 --fp4_exp_bits 2 --fp4_mantissa_bits 1 --w_group_size 128 --w_symmetric --eval_mode ppl --datasets wikitext ptb c4 --output_file cuda4.json
# 12.6 ours(FP4E1M2到3bit尾数，分离outlier)
CUDA_VISIBLE_DEVICES=4 python Iron_weight_only_quant/main.py --model_path /home/data/meta-llama/opt/6.7b/ --w_format fp4 --w_bits 4 --fp4_exp_bits 1 --fp4_mantissa_bits 2 --w_group_size 128 --w_symmetric --approximate --quant_dim 1 --eval_mode ppl --datasets wikitext ptb c4 --output_file cuda4.json \
--fp4_hi_align_start 0 --fp4_hi_align_exp_field 1 --fp4_tail_pad_bits 0
# 12.7 ours(FP4E2M1到3bit尾数，分离outlier)
CUDA_VISIBLE_DEVICES=4 python Iron_weight_only_quant/main.py --model_path /home/data/meta-llama/opt/6.7b/ --w_format fp4 --w_bits 4 --fp4_exp_bits 2 --fp4_mantissa_bits 1 --w_group_size 128 --w_symmetric --approximate --quant_dim 1 --eval_mode ppl --datasets wikitext ptb c4 --output_file cuda4.json \
--fp4_hi_align_start 0 --fp4_hi_align_exp_field 2 --fp4_tail_pad_bits 1
# 12.8 ours(FP4E2M1到3bit尾数，double近似）
CUDA_VISIBLE_DEVICES=4 python Iron_weight_only_quant/main.py --model_path /home/data/meta-llama/opt/6.7b/ --w_format fp4 --w_bits 4 --fp4_exp_bits 2 --fp4_mantissa_bits 1 --w_group_size 128 --w_symmetric --approximate --quant_dim 1 --eval_mode ppl --datasets wikitext ptb c4 --output_file cuda4.json \
--fp4_hi_align_start 0 --fp4_hi_align_exp_field 2 --fp4_tail_pad_bits 1 --double_approximate
# 12.9 ours(FP6E2M3到3bit尾数，分离outlier)
CUDA_VISIBLE_DEVICES=4 python Iron_weight_only_quant/main.py --model_path /home/data/meta-llama/opt/6.7b/ --w_format fp6 --w_bits 6 --fp6_exp_bits 2 --fp6_mantissa_bits 3 --w_group_size 128 --w_symmetric --approximate --quant_dim 1 --eval_mode ppl --datasets wikitext ptb c4 --output_file cuda4.json \
--fp6_hi_align_start 0 --fp6_hi_align_exp_field 2 --fp6_tail_pad_bits -1
# 12.10 ours(FP6E2M3到3bit尾数，double近似）
CUDA_VISIBLE_DEVICES=4 python Iron_weight_only_quant/main.py --model_path /home/data/meta-llama/opt/6.7b/ --w_format fp6 --w_bits 6 --fp6_exp_bits 2 --fp6_mantissa_bits 3 --w_group_size 128 --w_symmetric --approximate --quant_dim 1 --eval_mode ppl --datasets wikitext ptb c4 --output_file cuda4.json \
--fp6_hi_align_start 0 --fp6_hi_align_exp_field 2 --fp6_tail_pad_bits -1 --double_approximate
# 12.11 ours(FP6E3M2到3bit尾数，分离outlier)
CUDA_VISIBLE_DEVICES=4 python Iron_weight_only_quant/main.py --model_path /home/data/meta-llama/opt/6.7b/ --w_format fp6 --w_bits 6 --fp6_exp_bits 3 --fp6_mantissa_bits 2 --w_group_size 128 --w_symmetric --approximate --quant_dim 1 --eval_mode ppl --datasets wikitext ptb c4 --output_file cuda4.json \
--fp6_hi_align_start 4 --fp6_hi_align_exp_field 6 --fp6_tail_pad_bits 0
# 12.12 ours(FP6E3M2到3bit尾数，double近似）
CUDA_VISIBLE_DEVICES=4 python Iron_weight_only_quant/main.py --model_path /home/data/meta-llama/opt/6.7b/ --w_format fp6 --w_bits 6 --fp6_exp_bits 3 --fp6_mantissa_bits 2 --w_group_size 128 --w_symmetric --approximate --quant_dim 1 --eval_mode ppl --datasets wikitext ptb c4 --output_file cuda4.json \
--fp6_hi_align_start 4 --fp6_hi_align_exp_field 6 --fp6_tail_pad_bits 0 --double_approximate
# 13.14 ours(FP8到3bit尾数，分离outlier)
CUDA_VISIBLE_DEVICES=4 python Iron_weight_only_quant/main.py --model_path /home/data/meta-llama/opt/6.7b/ --w_format fp8 --w_bits 8 --w_group_size 128 --w_symmetric --approximate --quant_dim 1 --eval_mode ppl --datasets wikitext ptb c4 --output_file cuda4.json \
--fp8_hi_align_start 13 --fp8_hi_align_exp_field 15 --fp8_tail_pad_bits -1
# 13.15 ours(FP8到3bit尾数, double近似)
CUDA_VISIBLE_DEVICES=4 python Iron_weight_only_quant/main.py --model_path /home/data/meta-llama/opt/6.7b/ --w_format fp8 --w_bits 8 --w_group_size 128 --w_symmetric --approximate --quant_dim 1 --eval_mode ppl --datasets wikitext ptb c4 --output_file cuda4.json \
--fp8_hi_align_start 13 --fp8_hi_align_exp_field 15 --fp8_tail_pad_bits -1 --double_approximate
#endregion