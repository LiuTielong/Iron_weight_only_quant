# 4 bit, 下游任务实验
# FP16
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python Iron_weight_only_quant/main.py --model_path /home/data/llama2/Llama-2-70b-hf/ --w_bits 16 --w_group_size 64 --w_symmetric --eval_mode lm_eval --tasks arc_easy arc_challenge boolq rte lambada hellaswag piqa --num_fewshot 0 --output_file llama2-70b/cuda1.json --batch_size 8
# INT4
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python Iron_weight_only_quant/main.py --model_path /home/data/llama2/Llama-2-70b-hf/ --w_bits 4 --w_group_size 64 --w_symmetric --eval_mode lm_eval --tasks arc_easy arc_challenge boolq rte lambada hellaswag piqa --num_fewshot 0 --output_file llama2-70b/cuda1.json --batch_size 8

# BFP4
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python Iron_weight_only_quant/main.py --model_path /home/data/llama2/Llama-2-70b-hf/ --w_format bfp --w_bits 4 --w_group_size 64 --w_symmetric --eval_mode lm_eval --tasks arc_easy arc_challenge boolq rte lambada hellaswag piqa --num_fewshot 0 --output_file llama2-70b/cuda1.json --batch_size 8


# (FP4E1M2到3bit尾数，分离outlier)
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python Iron_weight_only_quant/main.py --model_path /home/data/llama2/Llama-2-70b-hf/ --w_format fp4 --w_bits 4 --fp4_exp_bits 1 --fp4_mantissa_bits 2 --w_group_size 64 --w_symmetric --approximate --quant_dim 1 --eval_mode lm_eval --tasks arc_easy arc_challenge boolq rte lambada hellaswag piqa --num_fewshot 0 --output_file llama2-70b/cuda1.json \
# --fp4_hi_align_start 0 --fp4_hi_align_exp_field 1 --fp4_tail_pad_bits 0 --batch_size 8

# (FP4E2M1到3bit尾数，double近似）
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python Iron_weight_only_quant/main.py --model_path /home/data/llama2/Llama-2-70b-hf/ --w_format fp4 --w_bits 4 --fp4_exp_bits 2 --fp4_mantissa_bits 1 --w_group_size 64 --w_symmetric --approximate --quant_dim 1 --eval_mode lm_eval --tasks arc_easy arc_challenge boolq rte lambada hellaswag piqa --num_fewshot 0 --output_file llama2-70b/cuda1.json \
# --fp4_hi_align_start 0 --fp4_hi_align_exp_field 2 --fp4_tail_pad_bits 1 --double_approximate --batch_size 8

# (FP6E2M3到3bit尾数，double近似）
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python Iron_weight_only_quant/main.py --model_path /home/data/llama2/Llama-2-70b-hf/ --w_format fp6 --w_bits 6 --fp6_exp_bits 2 --fp6_mantissa_bits 3 --w_group_size 64 --w_symmetric --approximate --quant_dim 1 --eval_mode lm_eval --tasks arc_easy arc_challenge boolq rte lambada hellaswag piqa --num_fewshot 0 --output_file llama2-70b/cuda1.json \
# --fp6_hi_align_start 0 --fp6_hi_align_exp_field 2 --fp6_tail_pad_bits -1 --double_approximate --batch_size 8

# (FP6E3M2到3bit尾数，double近似）
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python Iron_weight_only_quant/main.py --model_path /home/data/llama2/Llama-2-70b-hf/ --w_format fp6 --w_bits 6 --fp6_exp_bits 3 --fp6_mantissa_bits 2 --w_group_size 64 --w_symmetric --approximate --quant_dim 1 --eval_mode lm_eval --tasks arc_easy arc_challenge boolq rte lambada hellaswag piqa --num_fewshot 0 --output_file llama2-70b/cuda1.json \
# --fp6_hi_align_start 4 --fp6_hi_align_exp_field 6 --fp6_tail_pad_bits 0 --double_approximate --batch_size 8

# (FP8E4M3到3bit尾数, double近似)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python Iron_weight_only_quant/main.py --model_path /home/data/llama2/Llama-2-70b-hf/ --w_format fp8 --w_bits 8 --w_group_size 64 --w_symmetric --approximate --quant_dim 1 --eval_mode lm_eval --tasks arc_easy arc_challenge boolq rte lambada hellaswag piqa --num_fewshot 0 --output_file llama2-70b/cuda1.json \
--fp8_hi_align_start 13 --fp8_hi_align_exp_field 15 --fp8_tail_pad_bits -1 --double_approximate --batch_size 8


# FP4E1M2
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python Iron_weight_only_quant/main.py --model_path /home/data/llama2/Llama-2-70b-hf/ --w_format fp4 --w_bits 4 --fp4_exp_bits 1 --fp4_mantissa_bits 2 --w_group_size 64 --w_symmetric --eval_mode lm_eval --tasks arc_easy arc_challenge boolq rte lambada hellaswag piqa --num_fewshot 0 --output_file llama2-70b/cuda1.json --batch_size 8

# FP4E2M1
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python Iron_weight_only_quant/main.py --model_path /home/data/llama2/Llama-2-70b-hf/ --w_format fp4 --w_bits 4 --fp4_exp_bits 2 --fp4_mantissa_bits 1 --w_group_size 64 --w_symmetric --eval_mode lm_eval --tasks arc_easy arc_challenge boolq rte lambada hellaswag piqa --num_fewshot 0 --output_file llama2-70b/cuda1.json --batch_size 8

# (FP4E2M1到3bit尾数，分离outlier)
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python Iron_weight_only_quant/main.py --model_path /home/data/llama2/Llama-2-70b-hf/ --w_format fp4 --w_bits 4 --fp4_exp_bits 2 --fp4_mantissa_bits 1 --w_group_size 64 --w_symmetric --approximate --quant_dim 1 --eval_mode lm_eval --tasks arc_easy arc_challenge boolq rte lambada hellaswag piqa --num_fewshot 0 --output_file llama2-70b/cuda1.json \
# --fp4_hi_align_start 0 --fp4_hi_align_exp_field 2 --fp4_tail_pad_bits 1 --batch_size 8
# (FP6E2M3到3bit尾数，分离outlier)
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python Iron_weight_only_quant/main.py --model_path /home/data/llama2/Llama-2-70b-hf/ --w_format fp6 --w_bits 6 --fp6_exp_bits 2 --fp6_mantissa_bits 3 --w_group_size 64 --w_symmetric --approximate --quant_dim 1 --eval_mode lm_eval --tasks arc_easy arc_challenge boolq rte lambada hellaswag piqa --num_fewshot 0 --output_file llama2-70b/cuda1.json \
# --fp6_hi_align_start 0 --fp6_hi_align_exp_field 2 --fp6_tail_pad_bits -1 --batch_size 8
# (FP6E3M2到3bit尾数，分离outlier)
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python Iron_weight_only_quant/main.py --model_path /home/data/llama2/Llama-2-70b-hf/ --w_format fp6 --w_bits 6 --fp6_exp_bits 3 --fp6_mantissa_bits 2 --w_group_size 64 --w_symmetric --approximate --quant_dim 1 --eval_mode lm_eval --tasks arc_easy arc_challenge boolq rte lambada hellaswag piqa --num_fewshot 0 --output_file llama2-70b/cuda1.json \
# --fp6_hi_align_start 4 --fp6_hi_align_exp_field 6 --fp6_tail_pad_bits 0 --batch_size 8
# (FP8E4M3到3bit尾数，分离outlier)
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python Iron_weight_only_quant/main.py --model_path /home/data/llama2/Llama-2-70b-hf/ --w_format fp8 --w_bits 8 --w_group_size 64 --w_symmetric --approximate --quant_dim 1 --eval_mode lm_eval --tasks arc_easy arc_challenge boolq rte lambada hellaswag piqa --num_fewshot 0 --output_file llama2-70b/cuda1.json \
# --fp8_hi_align_start 13 --fp8_hi_align_exp_field 15 --fp8_tail_pad_bits -1 --batch_size 8