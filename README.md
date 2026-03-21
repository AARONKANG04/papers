# Papers

From-scratch implementations of foundational machine learning papers.

Implementations are primarily in Python/PyTorch, with C++ for low-level work like Flash Attention. Each subdirectory contains a self-contained reimplementation with training scripts, benchmarks, and personal `.md` notes where I break down and learn the key concepts from each paper.

## Implementations

| Paper | Directory | Description |
|-------|-----------|-------------|
| [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) (Kingma & Ba, 2014) | [`adam/`](adam/) | Custom Adam optimizer benchmarked against SGD with momentum on CIFAR-10 using ResNet-18 |

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> **Blackwell GPUs (RTX 5090, etc.):** Stable PyTorch does not yet support Blackwell. You'll need the nightly build:
> ```bash
> pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
> ```

## License

MIT
