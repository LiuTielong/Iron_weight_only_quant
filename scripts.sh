# 运行最基础的weight-only量化，per-tensor，w_bits分别设置为16, 8, 4
CUDA_VISIBLE_DEVICES=2 python main.py --model_path /home/data/meta-llama/opt/6.7b/ --w_bits 16 8 4 --w_group_size -1 --datasets wikitext

# 运行基础的weight-only量化，per-group=128, w_bits设置为4，做对称量化.
CUDA_VISIBLE_DEVICES=2 python main.py --model_path /home/data/meta-llama/opt/6.7b/ --w_bits 4 --w_group_size 128 --w_symmetric --datasets wikitext

# 运行基础的weight-only量化，但是推理引擎选择FIGLUT-I
python main.py --model_path /home/data/meta-llama/opt/125m/ --w_bits 4 --w_group_size 128 --datasets wikitext ptb c4 --mode 2 --w_symmetric

# 使用GPTQ进行weight-only量化，采用WikiText2校准样本
CUDA_VISIBLE_DEVICES=0 python main.py --model_path /home/data/meta-llama/opt/6.7b/ --w_bits 4 --w_group_size -2 --gptq --nsamples 128 --datasets wikitext ptb c4 --w_symmetric

# 使用GPTQ进行per-group量化 (group size = 128)
CUDA_VISIBLE_DEVICES=0 python main.py --model_path /home/data/meta-llama/opt/6.7b/ --w_bits 4 --w_group_size 128 --gptq --nsamples 128 --datasets wikitext ptb c4 --w_symmetric

# 使用GPTQ量化的原始代码进行weight-only量化
CUDA_VISIBLE_DEVICES=2 python gptq/opt.py --model /home/data/meta-llama/opt/6.7b/ --dataset wikitext2 --seed 0 --wbits 16 --groupsize 128 --sym

# 将权重量化为BCQ格式
CUDA_VISIBLE_DEVICES=5 python bcq/quantize_rtn_to_bcq.py --output_dir /home/liutielong/Files_2025/LLM_quantization/BCQ_weights/ \
--model_name_or_path /home/data/meta-llama/opt/125m/ --qbits 4 --group_size 128 \
--pack_binary --verify_reconstruction

# 将权重使用RTN量化到FP4数据格式
CUDA_VISIBLE_DEVICES=4 python main.py --model_path /home/data/meta-llama/opt/6.7b/ --w_format fp4 --w_bits 4 --w_group_size 128 --w_symmetric --datasets wikitext
CUDA_VISIBLE_DEVICES=4 python main.py --model_path /home/data/meta-llama/opt/125m/ --w_format fp4 --w_bits 4 --w_group_size 128 --w_symmetric --datasets wikitext



# 近似量化
CUDA_VISIBLE_DEVICES=0 python main.py --model_path /home/data/meta-llama/opt/6.7b/ --w_format fp8 --w_bits 8 --w_group_size -1 --datasets wikitext --w_symmetric 
CUDA_VISIBLE_DEVICES=1 python main.py --model_path /home/data/meta-llama/opt/6.7b/ --w_format fp8 --w_bits 8 --w_group_size 128 --datasets wikitext --w_symmetric --approximate


# 做lm_eval测试的脚本
CUDA_VISIBLE_DEVICES=2 python Iron_weight_only_quant/eval_quant_lm_eval.py   --model_path /home/data/meta-llama/opt/6.7b   --tasks boolq    --w_bit 8 --w_group_size 128 \
--w_format fp8 --approximate   --batch_size 1 --device cuda   --offline   --hf_cache /home/liutielong/.cache/huggingface --quant_dim 0


CUDA_VISIBLE_DEVICES=0 python main.py --model_path /home/data/meta-llama/opt/6.7b/ --w_format fp8 --w_bits 8 --w_group_size 128 --datasets wikitext --w_symmetric --approximate --eval_mode ppl




CUDA_VISIBLE_DEVICES=0 python Iron_weight_only_quant/main.py --model_path /home/data/meta-llama/opt/6.7b/ --w_format fp8 --w_bits 8 --w_group_size 128 --tasks boolq --w_symmetric --approximate


# 1. 测量ppl
# 1.1 FP16模型
CUDA_VISIBLE_DEVICES=0 python Iron_weight_only_quant/main.py --model_path /home/data/meta-llama/opt/6.7b/ --w_bits 16 --eval_mode ppl --datasets wikitext ptb c4
# 1.2 FP8模型
CUDA_VISIBLE_DEVICES=0 python Iron_weight_only_quant/main.py --model_path /home/data/meta-llama/opt/6.7b/ --w_format fp8 --w_bits 8 --w_group_size 128 --w_symmetric --eval_mode ppl --datasets wikitext ptb c4
# 1.3 FP8模型，近似量化，4bit尾数，不分离outlier
CUDA_VISIBLE_DEVICES=0 python Iron_weight_only_quant/main.py --model_path /home/data/meta-llama/opt/6.7b/ --w_format fp8 --w_bits 8 --w_group_size 128 --w_symmetric --approximate --quant_dim 1 --eval_mode ppl --datasets wikitext ptb c4 \
--fp8_hi_align_start 0 --fp8_hi_align_exp_field 15 --fp8_tail_pad_bits 0
# 1.4 FP8模型，近似量化，4bit尾数,分离outlier
CUDA_VISIBLE_DEVICES=1 python Iron_weight_only_quant/main.py --model_path /home/data/meta-llama/opt/6.7b/ --w_format fp8 --w_bits 8 --w_group_size 128 --w_symmetric --approximate --quant_dim 1 --eval_mode ppl --datasets wikitext ptb c4 \
--fp8_hi_align_start 12 --fp8_hi_align_exp_field 15 --fp8_tail_pad_bits 0
# 1.5 FP8模型，近似量化，5bit尾数，不分离outlier
CUDA_VISIBLE_DEVICES=2 python Iron_weight_only_quant/main.py --model_path /home/data/meta-llama/opt/6.7b/ --w_format fp8 --w_bits 8 --w_group_size 128 --w_symmetric --approximate --quant_dim 1 --eval_mode ppl --datasets wikitext ptb c4 \
--fp8_hi_align_start 0 --fp8_hi_align_exp_field 15 --fp8_tail_pad_bits 1
# 1.6 FP8模型，近似量化，5bit尾数,分离outlier
CUDA_VISIBLE_DEVICES=3 python Iron_weight_only_quant/main.py --model_path /home/data/meta-llama/opt/6.7b/ --w_format fp8 --w_bits 8 --w_group_size 128 --w_symmetric --approximate --quant_dim 1 --eval_mode ppl --datasets wikitext ptb c4 \
--fp8_hi_align_start 12 --fp8_hi_align_exp_field 15 --fp8_tail_pad_bits 1
# 1.7 FP8模型，双近似量化，4bit尾数
CUDA_VISIBLE_DEVICES=0 python Iron_weight_only_quant/main.py --model_path /home/data/meta-llama/opt/6.7b/ --w_format fp8 --w_bits 8 --w_group_size 128 --w_symmetric --approximate --quant_dim 1 --eval_mode ppl --datasets wikitext ptb c4 \
--fp8_hi_align_start 12 --fp8_hi_align_exp_field 15 --fp8_tail_pad_bits 0 --double_approximate
# 1.8 FP8模型，双近似量化，5bit尾数
CUDA_VISIBLE_DEVICES=1 python Iron_weight_only_quant/main.py --model_path /home/data/meta-llama/opt/6.7b/ --w_format fp8 --w_bits 8 --w_group_size 128 --w_symmetric --approximate --quant_dim 1 --eval_mode ppl --datasets wikitext ptb c4 \
--fp8_hi_align_start 12 --fp8_hi_align_exp_field 15 --fp8_tail_pad_bits 1 --double_approximate
# 1.9 BFP量化，总共5bit（4bit尾数）
CUDA_VISIBLE_DEVICES=0 python Iron_weight_only_quant/main.py --model_path /home/data/meta-llama/opt/6.7b/ --w_format bfp --w_bits 5 --w_group_size 128 --w_symmetric --eval_mode ppl --datasets wikitext ptb c4
# 1.10 BFP量化，总共6bit（5bit尾数）
CUDA_VISIBLE_DEVICES=1 python Iron_weight_only_quant/main.py --model_path /home/data/meta-llama/opt/6.7b/ --w_format bfp --w_bits 6 --w_group_size 128 --w_symmetric --eval_mode ppl --datasets wikitext ptb c4
# 1.11 FP8模型，近似量化，3bit尾数，不分离outlier
CUDA_VISIBLE_DEVICES=0 python Iron_weight_only_quant/main.py --model_path /home/data/meta-llama/opt/6.7b/ --w_format fp8 --w_bits 8 --w_group_size 128 --w_symmetric --approximate --quant_dim 1 --eval_mode ppl --datasets wikitext ptb c4 \
--fp8_hi_align_start 0 --fp8_hi_align_exp_field 15 --fp8_tail_pad_bits -1
# 1.12 FP8模型，近似量化，3bit尾数,分离outlier
CUDA_VISIBLE_DEVICES=1 python Iron_weight_only_quant/main.py --model_path /home/data/meta-llama/opt/6.7b/ --w_format fp8 --w_bits 8 --w_group_size 128 --w_symmetric --approximate --quant_dim 1 --eval_mode ppl --datasets wikitext ptb c4 \
--fp8_hi_align_start 13 --fp8_hi_align_exp_field 15 --fp8_tail_pad_bits -1
# 1.13 FP8模型，双近似量化，3bit尾数
CUDA_VISIBLE_DEVICES=0 python Iron_weight_only_quant/main.py --model_path /home/data/meta-llama/opt/6.7b/ --w_format fp8 --w_bits 8 --w_group_size 128 --w_symmetric --approximate --quant_dim 1 --eval_mode ppl --datasets wikitext ptb c4 \
--fp8_hi_align_start 13 --fp8_hi_align_exp_field 15 --fp8_tail_pad_bits -1 --double_approximate


# 2. 测量acc
# 2.1 FP16模型
CUDA_VISIBLE_DEVICES=3 python Iron_weight_only_quant/main.py --model_path /home/data/meta-llama/opt/6.7b/ --w_bits 16 --eval_mode lm_eval  --tasks arc_easy arc_challenge boolq gsm8k rte lambada hellaswag piqa --num_fewshot 0
CUDA_VISIBLE_DEVICES=3 python Iron_weight_only_quant/main.py --model_path /home/data/meta-llama/opt/6.7b/ --w_bits 16 --eval_mode lm_eval  --tasks arc_easy gsm8k --num_fewshot 5
# 2.2 FP8模型
CUDA_VISIBLE_DEVICES=2 python Iron_weight_only_quant/main.py --model_path /home/data/meta-llama/opt/6.7b/ --w_format fp8 --w_bits 8 --w_group_size 128 --w_symmetric --eval_mode lm_eval  --tasks arc_easy arc_challenge boolq rte lambada hellaswag piqa --num_fewshot 0
# 2.3 FP8模型，近似量化，4bit尾数，不分离outlier
CUDA_VISIBLE_DEVICES=3 python Iron_weight_only_quant/main.py --model_path /home/data/meta-llama/opt/6.7b/ --w_format fp8 --w_bits 8 --w_group_size 128 --w_symmetric --approximate --quant_dim 1 --eval_mode lm_eval --tasks arc_easy arc_challenge boolq rte lambada hellaswag piqa --num_fewshot 0 \
--fp8_hi_align_start 0 --fp8_hi_align_exp_field 15 --fp8_tail_pad_bits 0
# 2.4 FP8模型，近似量化，4bit尾数,分离outlier
CUDA_VISIBLE_DEVICES=4 python Iron_weight_only_quant/main.py --model_path /home/data/meta-llama/opt/6.7b/ --w_format fp8 --w_bits 8 --w_group_size 128 --w_symmetric --approximate --quant_dim 1 --eval_mode lm_eval --tasks arc_easy arc_challenge boolq rte lambada hellaswag piqa --num_fewshot 0 \
--fp8_hi_align_start 12 --fp8_hi_align_exp_field 15 --fp8_tail_pad_bits 0
# 2.5 FP8模型，近似量化，5bit尾数，不分离outlier
CUDA_VISIBLE_DEVICES=5 python Iron_weight_only_quant/main.py --model_path /home/data/meta-llama/opt/6.7b/ --w_format fp8 --w_bits 8 --w_group_size 128 --w_symmetric --approximate --quant_dim 1 --eval_mode lm_eval --tasks arc_easy arc_challenge boolq rte lambada hellaswag piqa --num_fewshot 0 \
--fp8_hi_align_start 0 --fp8_hi_align_exp_field 15 --fp8_tail_pad_bits 1
# 2.6 FP8模型，近似量化，5bit尾数,分离outlier
CUDA_VISIBLE_DEVICES=6 python Iron_weight_only_quant/main.py --model_path /home/data/meta-llama/opt/6.7b/ --w_format fp8 --w_bits 8 --w_group_size 128 --w_symmetric --approximate --quant_dim 1 --eval_mode lm_eval --tasks arc_easy arc_challenge boolq rte lambada hellaswag piqa --num_fewshot 0 \
--fp8_hi_align_start 12 --fp8_hi_align_exp_field 15 --fp8_tail_pad_bits 1
# 2.7 FP8模型，双近似量化，4bit尾数
CUDA_VISIBLE_DEVICES=0 python Iron_weight_only_quant/main.py --model_path /home/data/meta-llama/opt/6.7b/ --w_format fp8 --w_bits 8 --w_group_size 128 --w_symmetric --approximate --quant_dim 1 --eval_mode lm_eval --tasks arc_easy arc_challenge boolq rte lambada hellaswag piqa --num_fewshot 0 \
--fp8_hi_align_start 12 --fp8_hi_align_exp_field 15 --fp8_tail_pad_bits 0 --double_approximate
# 2.8 FP8模型，双近似量化，5bit尾数
CUDA_VISIBLE_DEVICES=1 python Iron_weight_only_quant/main.py --model_path /home/data/meta-llama/opt/6.7b/ --w_format fp8 --w_bits 8 --w_group_size 128 --w_symmetric --approximate --quant_dim 1 --eval_mode lm_eval --tasks arc_easy arc_challenge boolq rte lambada hellaswag piqa --num_fewshot 0 \
--fp8_hi_align_start 12 --fp8_hi_align_exp_field 15 --fp8_tail_pad_bits 1 --double_approximate
# 2.9 BFP量化，总共5bit（4bit尾数）
CUDA_VISIBLE_DEVICES=0 python Iron_weight_only_quant/main.py --model_path /home/data/meta-llama/opt/6.7b/ --w_format bfp --w_bits 5 --w_group_size 128 --w_symmetric --eval_mode lm_eval  --tasks arc_easy arc_challenge boolq rte lambada hellaswag piqa --num_fewshot 0
# 2.10 BFP量化，总共6bit（5bit尾数）
CUDA_VISIBLE_DEVICES=1 python Iron_weight_only_quant/main.py --model_path /home/data/meta-llama/opt/6.7b/ --w_format bfp --w_bits 6 --w_group_size 128 --w_symmetric --eval_mode lm_eval  --tasks arc_easy arc_challenge boolq rte lambada hellaswag piqa --num_fewshot 0
# 2.11 FP8模型，近似量化，3bit尾数，不分离outlier
CUDA_VISIBLE_DEVICES=2 python Iron_weight_only_quant/main.py --model_path /home/data/meta-llama/opt/6.7b/ --w_format fp8 --w_bits 8 --w_group_size 128 --w_symmetric --approximate --quant_dim 1 --eval_mode lm_eval --tasks arc_easy arc_challenge boolq rte lambada hellaswag piqa --num_fewshot 0 \
--fp8_hi_align_start 0 --fp8_hi_align_exp_field 15 --fp8_tail_pad_bits -1
# 2.12 FP8模型，近似量化，3bit尾数,分离outlier
CUDA_VISIBLE_DEVICES=3 python Iron_weight_only_quant/main.py --model_path /home/data/meta-llama/opt/6.7b/ --w_format fp8 --w_bits 8 --w_group_size 128 --w_symmetric --approximate --quant_dim 1 --eval_mode lm_eval --tasks arc_easy arc_challenge boolq rte lambada hellaswag piqa --num_fewshot 0 \
--fp8_hi_align_start 13 --fp8_hi_align_exp_field 15 --fp8_tail_pad_bits -1
# 2.13 FP8模型，双近似量化，3bit尾数
CUDA_VISIBLE_DEVICES=4 python Iron_weight_only_quant/main.py --model_path /home/data/meta-llama/opt/6.7b/ --w_format fp8 --w_bits 8 --w_group_size 128 --w_symmetric --approximate --quant_dim 1 --eval_mode lm_eval --tasks arc_easy arc_challenge boolq rte lambada hellaswag piqa --num_fewshot 0 \
--fp8_hi_align_start 13 --fp8_hi_align_exp_field 15 --fp8_tail_pad_bits -1 --double_approximate



# 3.FP6的PPL实验
# 3.1 FP6模型
CUDA_VISIBLE_DEVICES=0 python Iron_weight_only_quant/main.py --model_path /home/data/meta-llama/opt/6.7b/ --w_format fp6 --w_bits 6 --w_group_size 128 --w_symmetric --eval_mode ppl --datasets wikitext ptb c4
# 3.2 FP6模型，近似量化，4bit尾数，不分离outlier
CUDA_VISIBLE_DEVICES=2 python Iron_weight_only_quant/main.py --model_path /home/data/meta-llama/opt/6.7b/ --w_format fp6 --w_bits 6 --w_group_size 128 --w_symmetric --approximate --quant_dim 1 --eval_mode ppl --datasets wikitext ptb c4 \
--fp6_hi_align_start 0 --fp8_hi_align_exp_field 7 --fp8_tail_pad_bits 1
# 3.3 FP6模型，近似量化，4bit尾数，分离outlier
CUDA_VISIBLE_DEVICES=3 python Iron_weight_only_quant/main.py --model_path /home/data/meta-llama/opt/6.7b/ --w_format fp6 --w_bits 6 --w_group_size 128 --w_symmetric --approximate --quant_dim 1 --eval_mode ppl --datasets wikitext ptb c4 \
--fp6_hi_align_start 4 --fp8_hi_align_exp_field 7 --fp8_tail_pad_bits 1
# 3.4 FP6模型，双近似量化，4bit尾数
CUDA_VISIBLE_DEVICES=4 python Iron_weight_only_quant/main.py --model_path /home/data/meta-llama/opt/6.7b/ --w_format fp6 --w_bits 6 --w_group_size 128 --w_symmetric --approximate --quant_dim 1 --eval_mode ppl --datasets wikitext ptb c4 \
--fp6_hi_align_start 4 --fp8_hi_align_exp_field 7 --fp8_tail_pad_bits 1 --double_approximate

# 4. FP6的ACC实验
# 4.1 FP6模型
CUDA_VISIBLE_DEVICES=1 python Iron_weight_only_quant/main.py --model_path /home/data/meta-llama/opt/6.7b/ --w_format fp6 --w_bits 6 --w_group_size 128 --w_symmetric --eval_mode lm_eval  --tasks arc_easy arc_challenge boolq rte lambada hellaswag piqa --num_fewshot 0
# 4.2 FP6模型，近似量化，4bit尾数，不分离outlier
CUDA_VISIBLE_DEVICES=5 python Iron_weight_only_quant/main.py --model_path /home/data/meta-llama/opt/6.7b/ --w_format fp6 --w_bits 6 --w_group_size 128 --w_symmetric --approximate --quant_dim 1 --eval_mode lm_eval --tasks arc_easy arc_challenge boolq rte lambada hellaswag piqa --num_fewshot 0 \
--fp8_hi_align_start 0 --fp8_hi_align_exp_field 7 --fp8_tail_pad_bits 1
# 4.3 FP6模型，近似量化，4bit尾数，分离outlier
CUDA_VISIBLE_DEVICES=6 python Iron_weight_only_quant/main.py --model_path /home/data/meta-llama/opt/6.7b/ --w_format fp6 --w_bits 6 --w_group_size 128 --w_symmetric --approximate --quant_dim 1 --eval_mode lm_eval --tasks arc_easy arc_challenge boolq rte lambada hellaswag piqa --num_fewshot 0 \
--fp8_hi_align_start 4 --fp8_hi_align_exp_field 7 --fp8_tail_pad_bits 1
# 4.4 FP6模型，双近似量化，4bit尾数
CUDA_VISIBLE_DEVICES=7 python Iron_weight_only_quant/main.py --model_path /home/data/meta-llama/opt/6.7b/ --w_format fp6 --w_bits 6 --w_group_size 128 --w_symmetric --approximate --quant_dim 1 --eval_mode lm_eval --tasks arc_easy arc_challenge boolq rte lambada hellaswag piqa --num_fewshot 0 \
--fp8_hi_align_start 4 --fp8_hi_align_exp_field 7 --fp8_tail_pad_bits 1 --double_approximate