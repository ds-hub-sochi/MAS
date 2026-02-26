# MAS: A Millennium of Arabic Manuscripts in Three Styles

**MAS (Medieval Arabic Script)** is a multi-domain line-level OCR benchmark for historical Arabic manuscripts spanning from the 10th to the 20th century. This repository contains the training and evaluation configurations used for benchmarking state-of-the-art OCR systems and Large Vision-Language Models (LVLMs) on this dataset.

## 📖 Overview

The MAS dataset aims to bridge the gap in Arabic OCR by providing authentic historical benchmarks. While existing datasets often feature modern scribed text under controlled conditions, MAS provides:
- **11,841 annotated lines** from historical manuscripts.
- **Three major calligraphic styles**: Naskh, Taliq, and Nastaliq.
- **Multiple scholarly domains**: Astronomy, History, Mathematics, Religion, and Sufi Literature.
- **Millennium coverage**: Documents dating from the 10th to the 20th centuries.

Our research demonstrates that while modern LVLMs (like GPT-4o and Gemini) show limited zero-shot performance on medieval scripts, they significantly outperform traditional OCR systems after parameter-efficient fine-tuning (SFT).

## 📂 Repository Structure

```text
.
├── configs/                # Training & evaluation configuration files
│   ├── easy_ocr_...        # EasyOCR G1 fine-tuning configs
│   ├── mmocr_...           # MMOCR ABINet configs for MAS & Muharaf
│   ├── paddle_...          # PaddleOCR Server/Mobile configs
├── README.md               # Project documentation
└── .gitignore              # Tracking only configs and docs
```

## 🛠️ Supported Frameworks

This repository provides optimized configurations for:
- **EasyOCR**: Lightweight CRNN-based model.
- **PaddleOCR (PP-OCRv4/v5)**: Production-ready pipeline with Server and Mobile variants.
- **MMOCR**: Modular framework featuring the ABINet architecture.

## 🚀 Key Findings

Our experimental analysis reveals that:
1. **LVLMs are promising**: After fine-tuning, compact open-source LVLMs surpass many traditional OCR tools on historical data.
2. **Domain Specificity Matters**: Supervision on authentic medieval manuscripts is crucial. Training on modern or synthetic data alone only partially addresses the gaps caused by temporal and stylistic shifts.
3. **Generalization**: The inclusion of diverse calligraphic styles in MAS ensures better model robustness across different archival hands.

## 📄 Citation

If you use the MAS dataset or these configurations in your research, please cite:

```bibtex
@article{mas2026ocr,
  title={A Millennium of Arabic Manuscripts in Three Styles: A Line-Level OCR Benchmark for Naskh, Taliq, and Nastaliq},
  author={TODO},
  journal={ICDAR},
  year={2026}
}
```

---
*Developed as part of the ICDAR 2026 competition research.*
