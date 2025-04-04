**The paper is currently under review. Upon completion of the peer-review process, the complete code will be released.**

# 🌟 MVSAF-Net
Multi-View-guided Scale-Aware Feature Fusion Network for Oral Malignant Ulcer Detection

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)

> **Note**  
> 📢 Technical details of this project are currently under paper review. Partial implementation code will be progressively disclosed based on the paper acceptance status. Stay tuned for updates.

## 📖 Project Overview
This project proposes a novel **MVSAF-Net** architecture for **oral malignant ulcer detection**. Key innovations include:
- Multi-view Encoding Module
- Scale-aware Attention Mechanism
- Mutual Information-based Hierarchical Semantic Feature Fusion

## 🧠 Network Architecture
![Network Architecture](image/network.jpg)

▷ The architecture consists of three core modules:
- **Module 1**: Multi-view Embedding
- **Module 2**: Scale-aware Attention Mechanism
- **Module 3**: Mutual Information Hierarchical Feature Fusion

## ⚙️ Environment Configuration
### Hardware Requirements
- NVIDIA GPU: ≥ RTX 4090 (24GB VRAM)
- RAM: ≥ 32GB
- Storage: ≥ 100GB SSD

### Software Dependencies
```bash
Python >= 3.8
PyTorch == 2.0.1
torchvision == 0.15.2
# Other dependencies...
