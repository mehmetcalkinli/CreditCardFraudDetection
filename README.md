# Credit Card Fraud Detection

A machine learning-based system for detecting fraudulent credit card transactions using advanced algorithms and data analysis techniques.

## 📋 Overview

This project implements machine learning models to detect fraudulent credit card transactions in real-time. It uses various features extracted from transaction data to identify potentially fraudulent activities while minimizing false positives.

## ✨ Features

- Real-time fraud detection
- Multiple machine learning models support
- Feature engineering and preprocessing
- Model performance evaluation metrics
- Easy-to-use API interface
- Scalable architecture

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip (Python package installer)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/CreditCardFraudDetection.git
cd CreditCardFraudDetection
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## 🛠️ Usage

1. Prepare your dataset:
   - Place your transaction data in the `data` directory
   - Ensure the data follows the required format

2. Train the model:
```bash
python train.py --data_path data/transactions.csv --model_type random_forest
```

3. Make predictions:
```bash
python predict.py --input_file data/new_transactions.csv --output_file predictions.csv
```

## 📊 Model Performance

The system currently supports the following models:
- Random Forest
- XGBoost
- LightGBM
- Neural Network

Performance metrics are evaluated using:
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

## 📁 Project Structure

```
CreditCardFraudDetection/
├── data/               # Data directory
├── models/            # Trained model files
├── src/               # Source code
│   ├── preprocessing/ # Data preprocessing modules
│   ├── models/       # Model implementations
│   └── utils/        # Utility functions
├── tests/            # Test files
├── requirements.txt  # Project dependencies
└── README.md        # Project documentation
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- Mehmet Çalkınlı, Baha Erdoğan


⭐ Star this repository if you find it useful!
