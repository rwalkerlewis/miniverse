# Local LLM Guide for Miniverse

This guide explains how to set up and use local LLMs with Miniverse instead of cloud-based API providers like OpenAI or Anthropic.

## Why Use Local LLMs?

- **Privacy**: Keep all simulation data on your own hardware
- **Cost**: No per-token API charges after initial setup
- **Experimentation**: Test different models without API rate limits
- **Offline**: Run simulations without internet connectivity
- **Control**: Fine-tune models for specific simulation domains

## Supported Local LLM Servers

Miniverse supports any OpenAI-compatible API server, including:

1. **Ollama** (Recommended for beginners)
2. **LM Studio**
3. **vLLM**
4. **LocalAI**
5. **text-generation-webui**
6. **llama.cpp server**

## Quick Start with Ollama (Recommended)

### 1. Install Ollama

**macOS / Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:**
Download from [ollama.com/download](https://ollama.com/download)

### 2. Download a Model

Ollama has a large library of models. Here are recommended models for Miniverse:

**For development/testing (faster, less accurate):**
```bash
# Llama 3.2 3B - Fast, good for testing
ollama pull llama3.2:3b

# Qwen 2.5 7B - Balanced speed/quality
ollama pull qwen2.5:7b
```

**For production simulations (slower, more accurate):**
```bash
# Llama 3.1 8B - Good general purpose
ollama pull llama3.1:8b

# Qwen 2.5 14B - Better reasoning
ollama pull qwen2.5:14b

# Llama 3.3 70B - Best quality (requires 40GB+ RAM)
ollama pull llama3.3:70b
```

**For JSON structured outputs (recommended for Miniverse):**
```bash
# Mistral models are excellent at structured outputs
ollama pull mistral:7b
ollama pull mistral-nemo:12b
```

### 3. Start Ollama Server

Ollama starts automatically after installation, but you can verify:

```bash
ollama serve
```

The server runs at `http://localhost:11434` by default.

### 4. Configure Miniverse

Set these environment variables:

```bash
export LLM_PROVIDER=openai
export LLM_MODEL=llama3.1:8b
export LOCAL_LLM_BASE_URL=http://localhost:11434/v1
export LOCAL_LLM_API_KEY=not-needed
```

Or create a `.env` file in your project root:

```env
LLM_PROVIDER=openai
LLM_MODEL=llama3.1:8b
LOCAL_LLM_BASE_URL=http://localhost:11434/v1
LOCAL_LLM_API_KEY=not-needed
```

### 5. Run Your Simulation

```bash
uv run python examples/workshop/run.py --llm --ticks 10
```

## LM Studio Setup

### 1. Install LM Studio

Download from [lmstudio.ai](https://lmstudio.ai/)

### 2. Download a Model

1. Open LM Studio
2. Click "Search" tab
3. Search for recommended models:
   - `TheBloke/Mistral-7B-Instruct-v0.2-GGUF`
   - `TheBloke/Llama-2-13B-chat-GGUF`
   - `TheBloke/OpenHermes-2.5-Mistral-7B-GGUF`
4. Download your preferred quantization (Q4_K_M is a good balance)

### 3. Start Local Server

1. Click "Local Server" tab in LM Studio
2. Select your downloaded model
3. Click "Start Server"
4. Note the server URL (usually `http://localhost:1234/v1`)

### 4. Configure Miniverse

```bash
export LLM_PROVIDER=openai
export LLM_MODEL=mistral-7b-instruct  # Use the model name from LM Studio
export LOCAL_LLM_BASE_URL=http://localhost:1234/v1
export LOCAL_LLM_API_KEY=not-needed
```

## vLLM Setup (Advanced)

vLLM is optimized for high-throughput inference on GPUs.

### 1. Install vLLM

```bash
pip install vllm
```

### 2. Start vLLM Server

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --port 8000
```

### 3. Configure Miniverse

```bash
export LLM_PROVIDER=openai
export LLM_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct
export LOCAL_LLM_BASE_URL=http://localhost:8000/v1
export LOCAL_LLM_API_KEY=not-needed
```

## Model Selection Guide

### For Miniverse Simulations

Different models have different strengths:

**Best for structured JSON outputs:**
- Mistral 7B Instruct
- Qwen 2.5 (any size)
- Llama 3.1 (any size)

**Best for creative social interactions:**
- OpenHermes 2.5
- Nous Hermes 2
- Llama 3.1 70B

**Best for fast iteration (development):**
- Llama 3.2 3B
- Phi-3 Mini
- Qwen 2.5 3B

**Best for complex reasoning:**
- Llama 3.3 70B
- Qwen 2.5 72B
- Mixtral 8x7B

### Quantization Levels

Models come in different quantization levels (compression):

- **Q8_0**: Highest quality, largest size
- **Q5_K_M**: Good balance (recommended)
- **Q4_K_M**: Fast, decent quality
- **Q3_K_M**: Fastest, lower quality
- **Q2_K**: Very fast, significant quality loss

For Miniverse, we recommend Q4_K_M or Q5_K_M for the best speed/quality balance.

## Hardware Requirements

### Minimum (for testing):
- 8GB RAM
- CPU-only inference
- Model: Llama 3.2 3B or Phi-3 Mini

### Recommended (for development):
- 16GB RAM
- NVIDIA GPU with 8GB+ VRAM (optional)
- Model: Llama 3.1 8B or Mistral 7B

### Production (for best results):
- 32GB+ RAM
- NVIDIA GPU with 24GB+ VRAM
- Model: Llama 3.3 70B or Qwen 2.5 72B

## Troubleshooting

### "Connection refused" errors

**Problem**: Miniverse can't connect to local LLM server

**Solutions**:
1. Verify server is running: `curl http://localhost:11434/v1/models`
2. Check correct port in `LOCAL_LLM_BASE_URL`
3. Ensure firewall allows local connections

### Slow inference

**Problem**: Actions take too long to generate

**Solutions**:
1. Use a smaller model (e.g., 3B instead of 70B)
2. Use more aggressive quantization (Q4 instead of Q8)
3. Enable GPU acceleration if available
4. Reduce `max_attempts` in retry logic

### JSON validation errors

**Problem**: Model outputs invalid JSON frequently

**Solutions**:
1. Use models known for good structured outputs (Mistral, Qwen)
2. Increase temperature slightly (0.7-0.9)
3. Update to latest model version
4. Try a larger model size

### Out of memory errors

**Problem**: System runs out of RAM/VRAM

**Solutions**:
1. Use a smaller model
2. Use more aggressive quantization
3. Close other applications
4. For Ollama: Reduce context window with `--ctx-size 2048`

## Performance Optimization

### Ollama Optimizations

```bash
# Reduce context window for faster inference
ollama run llama3.1:8b --ctx-size 2048

# Adjust thread count for CPU
ollama run llama3.1:8b --threads 8

# GPU acceleration (automatic if GPU detected)
ollama run llama3.1:8b
```

### Batch Processing

For Monte Carlo simulations with local LLMs, consider:

```bash
# Run fewer ticks per trial, more trials
uv run python examples/workshop/monte_carlo.py --runs 200 --ticks 10

# Instead of many ticks per trial
# uv run python examples/workshop/monte_carlo.py --runs 50 --ticks 40
```

### Caching Prompts

Ollama and other servers cache prompts automatically. Restarting your simulation multiple times will be faster than the first run.

## Example: Running Workshop with Local LLM

```bash
# 1. Start Ollama (if not running)
ollama serve

# 2. Pull a model
ollama pull llama3.1:8b

# 3. Set environment
export LLM_PROVIDER=openai
export LLM_MODEL=llama3.1:8b
export LOCAL_LLM_BASE_URL=http://localhost:11434/v1
export OPENAI_API_KEY=not-needed

# 4. Run with debugging to verify
DEBUG_LLM=true uv run python examples/workshop/run.py --llm --ticks 3

# 5. Run full simulation
uv run python examples/workshop/run.py --llm --ticks 20
```

## Example: Running Smallville with Local LLM

```bash
# Use a larger model for complex social simulation
ollama pull llama3.1:70b

export LLM_PROVIDER=openai
export LLM_MODEL=llama3.1:70b
export LOCAL_LLM_BASE_URL=http://localhost:11434/v1
export OPENAI_API_KEY=not-needed

uv run python examples/smallville/valentines_party.py
```

## Switching Between Local and Cloud

You can easily switch between local and cloud LLMs:

**Use OpenAI:**
```bash
export LLM_PROVIDER=openai
export LLM_MODEL=gpt-5-nano
export OPENAI_API_KEY=sk-your-key
unset LOCAL_LLM_BASE_URL  # Remove local override
```

**Use Local Ollama:**
```bash
export LLM_PROVIDER=openai
export LLM_MODEL=llama3.1:8b
export LOCAL_LLM_BASE_URL=http://localhost:11434/v1
export OPENAI_API_KEY=not-needed
```

**Use Anthropic:**
```bash
export LLM_PROVIDER=anthropic
export LLM_MODEL=claude-sonnet-4-5
export ANTHROPIC_API_KEY=sk-ant-your-key
unset LOCAL_LLM_BASE_URL  # Remove local override
```

## Advanced: Custom Model Fine-tuning

For domain-specific simulations, you can fine-tune models:

1. Collect simulation traces from runs
2. Format as training data (prompt/completion pairs)
3. Fine-tune with Ollama or HuggingFace
4. Load custom model in local server

See [docs/RESEARCH.md](RESEARCH.md) for references on fine-tuning for agent simulations.

## Resources

- **Ollama Documentation**: [github.com/ollama/ollama](https://github.com/ollama/ollama)
- **LM Studio Guide**: [lmstudio.ai/docs](https://lmstudio.ai/docs)
- **vLLM Documentation**: [docs.vllm.ai](https://docs.vllm.ai)
- **Model Leaderboards**: [huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
- **Ollama Model Library**: [ollama.com/library](https://ollama.com/library)

## Support

If you encounter issues with local LLMs:

1. Check this guide's troubleshooting section
2. Review `DEBUG_LLM=true` output for errors
3. Test your local server with `curl`:
   ```bash
   curl http://localhost:11434/v1/models
   ```
4. Open an issue at [github.com/miniverse-ai/miniverse/issues](https://github.com/miniverse-ai/miniverse/issues)
