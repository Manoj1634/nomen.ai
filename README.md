# 🌱 Creative Domain Name Generator (Mistral-7B)

A Streamlit web application that generates creative and brandable domain names using the Mistral-7B-Instruct model. This tool helps entrepreneurs and businesses find the perfect domain name for their ventures.

## Features

- **AI-Powered Generation**: Uses Mistral-7B-Instruct v0.3 model for creative domain suggestions
- **4-bit Quantization**: Optimized for memory efficiency while maintaining quality
- **Interactive Web Interface**: Clean Streamlit UI for easy interaction
- **Batch Processing**: Generate multiple domain suggestions at once
- **Export Functionality**: Download all suggestions as CSV
- **Session Management**: Track and review all generated suggestions

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Manoj1634/nomen.ai.git
cd nomen.ai
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit application:
```bash
streamlit run code/streamlit_base.py
```

2. Open your browser and navigate to the provided URL (usually `http://localhost:8501`)

3. Enter a business description in the text input field

4. Click "Generate Domain Names" to get AI-powered suggestions

5. Review and download your suggestions as needed

## Configuration

The application uses the following default settings (configurable in the sidebar):
- **Temperature**: 0.7 (controls creativity)
- **Top-k**: 50 (limits vocabulary diversity)
- **Top-p**: 0.90 (nucleus sampling)
- **Max Tokens**: 350 (response length)

## Model Details

- **Model**: Mistral-7B-Instruct-v0.3
- **Quantization**: 4-bit with double quantization
- **Compute Type**: bfloat16
- **Device**: Auto (CPU/GPU)

## File Structure

```
nomen.ai/
├── code/
│   ├── streamlit_base.py    # Main Streamlit application
│   ├── base.py             # Base model implementation
│   └── all_suggestions.csv # Generated suggestions (auto-created)
├── data/                   # Data directory
├── venv/                   # Virtual environment (excluded from git)
├── README.md              # This file
├── .gitignore             # Git ignore rules
└── requirements.txt       # Python dependencies
```

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended for faster inference)
- 8GB+ RAM (16GB+ recommended)

## Dependencies

- streamlit
- transformers
- torch
- pandas
- bitsandbytes (for quantization)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Mistral AI for the Mistral-7B-Instruct model
- Hugging Face for the transformers library
- Streamlit for the web framework 