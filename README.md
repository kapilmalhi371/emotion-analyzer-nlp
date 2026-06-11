# 🎭 Multimodal Emotion Analyzer

A multimodal NLP project that detects human emotions from **text** and **facial images** using state-of-the-art deep learning models — all wrapped in an interactive Streamlit web app.

---

## ✨ Features

- **Text Emotion Detection** — Classifies emotions in any input sentence using a fine-tuned BERT model (`nateraw/bert-base-uncased-emotion`)
- **Face Emotion Detection** — Analyzes facial expressions from uploaded images using DeepFace
- **Interactive UI** — Clean Streamlit interface, no coding required to use
- **Custom Model Training** — Includes a training script to fine-tune DistilBERT on the GoEmotions dataset

---

## 🧠 Tech Stack

| Layer | Tools |
|---|---|
| NLP Model | `transformers` (BERT / DistilBERT) |
| Face Analysis | `DeepFace` |
| Dataset | GoEmotions (simplified, 6 labels) |
| Training | HuggingFace `Trainer` API |
| UI | `Streamlit` |
| Image Processing | `Pillow`, `NumPy` |

---

## 📁 Project Structure

```
NLP Project/
├── code/
│   ├── emotion_app.py       # Streamlit web app
│   ├── train_model.py       # Model fine-tuning script
│   ├── audio_model.h5       # Saved Keras model
│   ├── saved_model/         # Fine-tuned DistilBERT model directory
│   └── requirements.txt     # Python dependencies
├── NLP Project Report.docx  # Full project report
└── Emotion-Analyzer.pptx    # Project presentation
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/emotion-analyzer-nlp.git
cd emotion-analyzer-nlp
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `deepface` and `torch` may require additional setup depending on your OS. A GPU is recommended for training but not required for inference.

### 3. Run the app

```bash
streamlit run emotion_app.py
```

### 4. (Optional) Retrain the model

```bash
python train_model.py
```

This fine-tunes DistilBERT on the GoEmotions dataset for 3 epochs and saves the model to `saved_model/`.

---

## 🎯 Emotion Labels

The text model detects the following emotions:

`anger` · `disgust` · `fear` · `joy` · `sadness` · `surprise`

---

## 📊 Model Details

- **Text:** Fine-tuned `distilbert-base-uncased` on [GoEmotions](https://huggingface.co/datasets/go_emotions) (simplified 6-class version). Evaluated with Accuracy and Weighted F1-Score.
- **Face:** `DeepFace` library with pre-trained facial expression recognition.

---

## 📸 Demo

| Text Analysis | Face Analysis |
|---|---|
| Input any sentence → get the dominant emotion with confidence score | Upload a face image → detect emotion from facial expression |

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🙋‍♂️ Author

Built as part of an NLP course project. Feel free to fork, star ⭐, and contribute!
