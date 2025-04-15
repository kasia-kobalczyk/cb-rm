VLLM_DISABLE_COMPILE_CACHE=1 vllm serve meta-llama/Llama-4-Scout-17B-16E-Instruct \
 --tensor-parallel-size 4 \
 --max-model-len 4096 --override-generation-config='{"attn_temperature_tuning": true}' \
 --download_dir /mnt/pdata/knk25/huggingface/hub/ 
 #--host localhost \
 #--port 8000 \
 