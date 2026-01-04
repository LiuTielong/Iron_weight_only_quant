# 5 bit, acc
# 但是将group size调大到128，看看INT量化会不会变差。
# INT5
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python Iron_weight_only_quant/main.py --model_path /home/data/llama2/Llama-2-70b-hf/ --w_bits 5 --w_group_size 128 --w_symmetric --eval_mode lm_eval --tasks arc_easy arc_challenge boolq rte lambada hellaswag piqa --num_fewshot 0 --batch_size 12 --output_file llama2-70b/cuda6.json
# BFP5
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python Iron_weight_only_quant/main.py --model_path /home/data/llama2/Llama-2-70b-hf/ --w_format bfp --w_bits 5 --w_group_size 128 --w_symmetric --eval_mode lm_eval --tasks arc_easy arc_challenge boolq rte lambada hellaswag piqa --num_fewshot 0 --batch_size 12 --output_file llama2-70b/cuda6.json

# FP8-E4M3 double近似，4bit尾数
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python Iron_weight_only_quant/main.py --model_path /home/data/llama2/Llama-2-70b-hf/ --w_format fp8 --w_bits 8 --fp8_exp_bits 4 --fp8_mantissa_bits 3 --w_group_size 128 --w_symmetric --approximate --quant_dim 1 --eval_mode lm_eval --tasks arc_easy arc_challenge boolq rte lambada hellaswag piqa --num_fewshot 0 --batch_size 12 \
--fp8_hi_align_start 12 --fp8_hi_align_exp_field 15 --fp8_tail_pad_bits 0 --double_approximate --output_file llama2-70b/cuda6.json

# FP8-E3M4 double近似，4bit尾数
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python Iron_weight_only_quant/main.py --model_path /home/data/llama2/Llama-2-70b-hf/ --w_format fp8 --w_bits 8 --fp8_exp_bits 3 --fp8_mantissa_bits 4 --w_group_size 128 --w_symmetric --approximate --quant_dim 1 --eval_mode lm_eval --tasks arc_easy arc_challenge boolq rte lambada hellaswag piqa --num_fewshot 0 --batch_size 12 \
--fp8_hi_align_start 4 --fp8_hi_align_exp_field 6 --fp8_tail_pad_bits -1 --double_approximate --output_file llama2-70b/cuda6.json

# FP8-E2M5 double近似，4bit尾数
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python Iron_weight_only_quant/main.py --model_path /home/data/llama2/Llama-2-70b-hf/ --w_format fp8 --w_bits 8 --fp8_exp_bits 2 --fp8_mantissa_bits 5 --w_group_size 128 --w_symmetric --approximate --quant_dim 1 --eval_mode lm_eval --tasks arc_easy arc_challenge boolq rte lambada hellaswag piqa --num_fewshot 0 --batch_size 12 \
--fp8_hi_align_start 0 --fp8_hi_align_exp_field 2 --fp8_tail_pad_bits -2 --double_approximate --output_file llama2-70b/cuda6.json

# ours(FP6E2M3到4bit尾数，double近似）
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python Iron_weight_only_quant/main.py --model_path /home/data/llama2/Llama-2-70b-hf/ --w_format fp6 --w_bits 6 --fp6_exp_bits 2 --fp6_mantissa_bits 3 --w_group_size 128 --w_symmetric --approximate --quant_dim 1 --eval_mode lm_eval --tasks arc_easy arc_challenge boolq rte lambada hellaswag piqa --num_fewshot 0 --batch_size 12 \
--fp6_hi_align_start 0 --fp6_hi_align_exp_field 2 --fp6_tail_pad_bits 0 --double_approximate --output_file llama2-70b/cuda6.json

# ours(FP6E3M2到4bit尾数，double近似）
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python Iron_weight_only_quant/main.py --model_path /home/data/llama2/Llama-2-70b-hf/ --w_format fp6 --w_bits 6 --fp6_exp_bits 3 --fp6_mantissa_bits 2 --w_group_size 128 --w_symmetric --approximate --quant_dim 1 --eval_mode lm_eval --tasks arc_easy arc_challenge boolq rte lambada hellaswag piqa --num_fewshot 0 --batch_size 12 \
--fp6_hi_align_start 4 --fp6_hi_align_exp_field 6 --fp6_tail_pad_bits 1 --double_approximate --output_file llama2-70b/cuda6.json

# FP8-E4M3近似，4bit尾数，分离out
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python Iron_weight_only_quant/main.py --model_path /home/data/llama2/Llama-2-70b-hf/ --w_format fp8 --w_bits 8 --fp8_exp_bits 4 --fp8_mantissa_bits 3 --w_group_size 128 --w_symmetric --approximate --quant_dim 1 --eval_mode lm_eval --tasks arc_easy arc_challenge boolq rte lambada hellaswag piqa --num_fewshot 0 --batch_size 12 \
# --fp8_hi_align_start 12 --fp8_hi_align_exp_field 15 --fp8_tail_pad_bits 0 --output_file llama2-70b/cuda6.json
# FP8-E3M4近似，4bit尾数，分离out
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python Iron_weight_only_quant/main.py --model_path /home/data/llama2/Llama-2-70b-hf/ --w_format fp8 --w_bits 8 --fp8_exp_bits 3 --fp8_mantissa_bits 4 --w_group_size 128 --w_symmetric --approximate --quant_dim 1 --eval_mode lm_eval --tasks arc_easy arc_challenge boolq rte lambada hellaswag piqa --num_fewshot 0 --batch_size 12 \
# --fp8_hi_align_start 4 --fp8_hi_align_exp_field 6 --fp8_tail_pad_bits -1 --output_file llama2-70b/cuda6.json
# FP8-E2M5近似，4bit尾数，分离out
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python Iron_weight_only_quant/main.py --model_path /home/data/llama2/Llama-2-70b-hf/ --w_format fp8 --w_bits 8 --fp8_exp_bits 2 --fp8_mantissa_bits 5 --w_group_size 128 --w_symmetric --approximate --quant_dim 1 --eval_mode lm_eval --tasks arc_easy arc_challenge boolq rte lambada hellaswag piqa --num_fewshot 0 --batch_size 12 \
# --fp8_hi_align_start 0 --fp8_hi_align_exp_field 2 --fp8_tail_pad_bits -2 --output_file llama2-70b/cuda6.json
# ours(FP6E2M3到4bit尾数，分离outlier)
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python Iron_weight_only_quant/main.py --model_path /home/data/llama2/Llama-2-70b-hf/ --w_format fp6 --w_bits 6 --fp6_exp_bits 2 --fp6_mantissa_bits 3 --w_group_size 128 --w_symmetric --approximate --quant_dim 1 --eval_mode lm_eval --tasks arc_easy arc_challenge boolq rte lambada hellaswag piqa --num_fewshot 0 --batch_size 12 \
# --fp6_hi_align_start 0 --fp6_hi_align_exp_field 2 --fp6_tail_pad_bits 0 --output_file llama2-70b/cuda6.json
# ours(FP6E3M2到4bit尾数，分离outlier)
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python Iron_weight_only_quant/main.py --model_path /home/data/llama2/Llama-2-70b-hf/ --w_format fp6 --w_bits 6 --fp6_exp_bits 3 --fp6_mantissa_bits 2 --w_group_size 128 --w_symmetric --approximate --quant_dim 1 --eval_mode lm_eval --tasks arc_easy arc_challenge boolq rte lambada hellaswag piqa --num_fewshot 0 --batch_size 12 \
# --fp6_hi_align_start 4 --fp6_hi_align_exp_field 6 --fp6_tail_pad_bits 1 --output_file llama2-70b/cuda6.json