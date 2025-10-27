🌸 Iris Flower Classification with PyTorch & MLflow

This project implements a lightweight deep learning workflow for classifying iris flower species using a Convolutional Neural Network (CNN). The pipeline includes dataset handling, model training, experiment tracking, and artifact logging via MLflow and DagsHub.

✅ Key Features
Feature	Description
CNN Model	Classifies 3 types of iris flowers
Custom Dataset	Uses TXT-based features and label storage
MLflow Logging	Tracks metrics, hyperparameters, and artifacts
DagsHub Integration	Remote experiment tracking & visualization
PyTorch Training	CPU acceleration supported
📂 Project Structure
project/
│
├─ app/
│  ├─ app.py              # Dataset loader & CNN model
│  └─ __init__.py
│
├─ mlflow_tracking.py     # Training + MLflow integration
├─ dataset/               # Auto-downloaded iris dataset
└─ train-result/          # (Optional) Training outputs

📦 Dependencies

Install required packages:

pip install torch torchvision mlflow dagshub

🔗 MLflow + DagsHub Setup

Set tracking server environment variables:

set MLFLOW_TRACKING_URI=https://dagshub.com/fanye55/final-project.mlflow
set MLFLOW_TRACKING_USERNAME=fanye55
set MLFLOW_TRACKING_PASSWORD=<YOUR-PERSONAL-ACCESS-TOKEN>


Or pass token via CLI:

python mlflow_tracking.py --experiment iris-cnn --token <YOUR-TOKEN>

🚀 Run Training with MLflow Logging
python mlflow_tracking.py --experiment iris-cnn


✅ The script will:

Download dataset automatically (if missing)

Train CNN for N epochs

Log metrics (loss, accuracy)

Upload trained model artifact

Display MLflow tracking URL

Example output:

Epoch 1/3 — loss=0.4708 acc=0.7929
Epoch 2/3 — loss=0.1475 acc=0.9443
Epoch 3/3 — loss=0.1232 acc=0.9514

Training complete. Logged to MLflow.

📈 View Training Results Online

🔍 MLflow UI at DagsHub:

➡️ https://dagshub.com/fanye55/final-project.mlflow/#/experiments

You can compare model runs, download trained models, and analyze performance trends.

🧩 Model Info
Property	Value
Input Shape	4 numeric features
Channels	1D Convolution
Output Classes	3 iris species
Accuracy	~95% after 3 epochs
✅ Features Under Development

✅ Confusion matrix visualization

✅ Test set evaluation logging

⬜ MLflow Model Registry support

⬜ CI/CD automation with GitHub Actions

⬜ Inference service deployment (FastAPI)

🤝 Contribution

Pull requests are welcome!
For major changes, please open an issue first to discuss proposed updates.

📜 License

MIT License — free for personal and commercial usage.

If you'd like, I can:

✅ Upload this README directly to your GitHub repo
✅ Add images / diagrams for better documentation
✅ Include badges (GitHub Actions, MLflow link, DagsHub repo status)

Would you like me to:

📌 Add screenshots of MLflow dashboard?
📌 Provide a Chinese version of the README as well?

Just let me know! 😄
