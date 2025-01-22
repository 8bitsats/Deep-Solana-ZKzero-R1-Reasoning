# DeepSolana: A DeepSeek R1 Zero Advanced Reasoning Solana Blockchain AI Model

DeepSolana is a specialized AI model fine-tuned on the DeepSeek-R1-Zero foundation model, designed specifically for the Solana blockchain ecosystem. This repository contains the training code, documentation, and research paper for the model.

## Project Structure

```
.
├── README.md                    # Project documentation
├── DeepSolanaZKzeroR1.md       # Research paper
├── requirements.txt             # Python dependencies
├── train_deepsolana.py         # Main training script
├── train_qa_model.py           # QA model training script
├── train_sagemaker.py          # SageMaker training script
└── scripts/
    └── inspect_dataset.py      # Dataset inspection utility
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure Hugging Face credentials:
```bash
# Set your Hugging Face token
export HF_TOKEN="your_token_here"
```

## Training

The model can be trained using either:

1. Local training:
```bash
python train_deepsolana.py
```

2. SageMaker training:
```bash
python train_sagemaker.py
```

## Model Details

- Base Model: DeepSeek-R1-Zero
- Training Data: 29,092 examples
- Specialization: Solana blockchain ecosystem
- Use Cases: Development, Trading, Analysis

For detailed information about the model architecture, training methodology, and evaluation metrics, please refer to [DeepSolanaZKzeroR1.md](DeepSolanaZKzeroR1.md).

## Model Access

The trained model is available on the Hugging Face Hub:
[ordlibrary/DeepSolana](https://huggingface.co/ordlibrary/DeepSolana)

## Citation

```bibtex
@software{deepsolana2025,
  author = {OrdLibrary},
  title = {DeepSolana: Advanced Solana Blockchain AI Assistant},
  year = {2025},
  publisher = {Hugging Face},
  url = {https://huggingface.co/ordlibrary/DeepSolana}
}
```

## License

This project is released under the same license as the base DeepSeek-R1-Zero model, with additional terms for the fine-tuning data and modifications.
# Deep-Solana-ZKzero-R1-Reasoning
