# 原始模型的ACC
# opt
CUDA_VISIBLE_DEVICES=4 python Iron_weight_only_quant/main.py --model_path /home/data/meta-llama/opt/6.7b/ --w_bits 16 --eval_mode lm_eval --tasks arc_easy arc_challenge boolq rte lambada hellaswag piqa --num_fewshot 0 --output_file cuda_ori.json

# llama
CUDA_VISIBLE_DEVICES=5 python Iron_weight_only_quant/main.py --model_path /home/data/meta-llama/llama2-7b/ --w_bits 16 --eval_mode lm_eval --tasks arc_easy arc_challenge boolq rte lambada hellaswag piqa --num_fewshot 0 --output_file cuda_ori.json
