MODEL_DIR=/data/user_data/jjiahe/models

vllm serve $MODEL_DIR/Qwen3/Qwen3-32B --port 8000 --tensor-parallel-size 8 --enable-reasoning --reasoning-parser deepseek_r1 --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' --max-model-len 131072