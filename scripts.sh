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




CUDA_VISIBLE_DEVICES=0 python main.py --model_path /home/data/meta-llama/opt/6.7b/ --w_format fp8 --w_bits 8 --w_group_size -1 --datasets wikitext --w_symmetric 
CUDA_VISIBLE_DEVICES=0 python main.py --model_path /home/data/meta-llama/opt/6.7b/ --w_format fp8 --w_bits 8 --w_group_size 128 --datasets wikitext --w_symmetric --approximate
