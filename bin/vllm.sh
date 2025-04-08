python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B-Instruct \
 --tensor-parallel-size 2 \
 --host localhost \
 --port 8000 \
 --download_dir /mnt/pdata/knk25/huggingface/hub/ 