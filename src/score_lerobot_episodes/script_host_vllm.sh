uv run vllm serve nvidia/Cosmos-Reason2-2B \
  --allowed-local-media-path "$(pwd)" \
  --max-model-len 32768 \
  --media-io-kwargs '{"video": {"num_frames": 16}}' \
  --port 8000