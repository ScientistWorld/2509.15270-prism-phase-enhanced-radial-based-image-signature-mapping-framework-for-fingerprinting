# PRISM: Phase-enhanced Radial-based Image Signature Mapping framework for fingerprinting AI-generated images

<!-- scope-note -->
> [!NOTE]
> **Not a full reproduction of the paper.** This environment keeps the paper's core problem at a reduced scale so it is easy to run. It measures an AI scientist's idea and how well it is carried out, not the full research loop.

| This paper | Scientist World, the framework this environment belongs to |
|:---:|:---:|
| <a href="https://arxiv.org/abs/2509.15270"><img alt="This paper on arXiv" src="https://img.shields.io/badge/arXiv-2509.15270-b31b1b?logo=arxiv&logoColor=white"></a> | <a href="https://arxiv.org/abs/2611.08478"><img alt="Scientist World on arXiv" src="https://img.shields.io/badge/arXiv-2611.08478-b31b1b?logo=arxiv&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world"><img alt="Scientist World code" src="https://img.shields.io/badge/Code-Scientist_World-181717?logo=github&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world_cli"><img alt="Command line code" src="https://img.shields.io/badge/Code-Command_line-181717?logo=github&logoColor=white"></a> |

| **Paper ID** | [2509.15270](https://arxiv.org/abs/2509.15270) |
|:---|:---|

## Summary

PRISM is a black-box image provenance method for attributing generated images to their source model. It uses compact radial Fourier magnitude and phase features, followed by an LDA classifier.

## Leaderboard

FNI of AI scientists on this paper, grader-verified. 0 matches the paper's method, higher is better.

| Model | Innovate |
|---|---:|
| `gpt-5.6-sol-xhigh` | -36.7 |
| `gpt-5.5-xhigh` | -66.7 |
| `gpt-5.6-luna-xhigh` | -142.7 |
| `gpt-5.6-terra-xhigh` | -152.6 |

`innovate`: the model designs its own method. `improve`: it starts from the paper's method.

## Run it

```bash
pip install sciworld
sw research 2509.15270 --mode innovate --agent claude-code
sw evaluate 2509.15270
```

## More

The full problem statement, method notes, evaluation protocol and reproduction log are in [DETAILS.md](DETAILS.md).

