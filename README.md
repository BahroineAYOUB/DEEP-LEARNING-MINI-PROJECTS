# Deep Learning — Travaux Pratiques 🧠

**Étudiant :** BARHOINE AYOUB  
**Filière :** Génie Informatique (FIGI) — 2ème Année  
**Établissement :** Faculté Scientifique et Technique (FST) — Settat  
**Année :** 2026  

---

## 📚 Contenu du Repository

| TP | Notebook | Sujet | Framework | Dataset |
|----|----------|-------|-----------|---------|
| TP1 | `TP1_FashionMNIST_CNN_TransferLearning.ipynb` | CNN & Transfer Learning | PyTorch | FashionMNIST (70K images) |
| TP2 | `TP2_Stacked_LSTM_Power_Consumption.ipynb` | Stacked LSTM — Séries temporelles | PyTorch | Household Power UCI (~2M lignes) |
| TP3 | `TP3_LSTM_FuelConsumption_TinyML.ipynb` | LSTM + TinyML | TensorFlow/Keras + TFLite | FuelConsumption (1067 véhicules) |
| TP4 | `TP4_Transformers_NLP_HuggingFace.ipynb` | Transformers & NLP | Hugging Face Transformers | SST-2, CoNLL-2003, SQuAD |

---

## 🔬 Détails par TP

### TP1 — CNN & Transfer Learning (FashionMNIST)
- **Modèle 1 :** CNN basique (`FashionMNISTModel`) — SGD, 2 epochs → **~29%**
- **Modèle 2 :** CNN amélioré (`FashionCNN`) — BatchNorm + Dropout + Adam → **~90%**
- **Modèle 3 :** Transfer Learning — AlexNet / ResNet18 / VGG16 pré-entraînés ImageNet
- **Évaluation :** classification_report, matrice de confusion, courbes Loss/Accuracy

### TP2 — Stacked LSTM (Consommation Électrique)
- **Architecture :** 3 couches LSTM × 128 unités + Dropout(0.3) + Dense(1)
- **Pipeline :** feature engineering cyclique (hour_sin/cos), MinMaxScaler, séquences glissantes SEQ=60
- **Résultats :** MAE=0.1823 kW | RMSE=0.2741 kW | **R²=0.9124**

### TP3 — LSTM + TinyML (Consommation Carburant)
- **Architecture :** 2× LSTM(12) + Dense(32, ReLU) + Dense(1) — 2 321 paramètres (9 KB)
- **Pipeline :** EDA (pairplot + heatmap Spearman), découpage 70/30, entraînement 300 epochs
- **TinyML :** Conversion Keras → TFLite → Header C++ (28 KB) pour Arduino/ESP32/STM32
- **Résultats :** MAE ~0.80 L/100km, pas d'overfitting

### TP4 — Transformers & NLP (Hugging Face)
- **Pipeline 1 :** Sentiment Analysis (DistilBERT-SST2) → NEGATIVE score=0.9015
- **Pipeline 2 :** NER (BERT-large-CoNLL03) → 10 entités : ORG, LOC, PER, MISC
- **Pipeline 3 :** Question Answering (DistilBERT-SQuAD) → réponse extraite score=0.631
- **Bonus :** text-generation (GPT-2), fill-mask (BERT), summarization (BART), zero-shot

---

## 🚀 Utilisation

### Google Colab (recommandé)
Cliquez sur le badge [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/) ou ouvrez directement chaque notebook depuis GitHub.

### Local
```bash
git clone https://github.com/<votre-username>/deep-learning-tps.git
cd deep-learning-tps
pip install -r requirements.txt
jupyter notebook
```

---

## 📦 Dépendances

```
# TP1 & TP2
torch>=2.0.0
torchvision>=0.15.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
pandas>=2.0.0
numpy>=1.24.0
tqdm

# TP3
tensorflow>=2.15.0
keras>=3.0.0

# TP4
transformers>=5.0.0
datasets>=4.0.0
accelerate
sentencepiece
```

---

## 📊 Résultats Synthèse

| Modèle | Tâche | Métrique | Valeur |
|--------|-------|---------|--------|
| FashionMNISTModel (SGD) | Classification | Accuracy | ~29% |
| FashionCNN (Adam) | Classification | Accuracy | ~90% |
| Stacked LSTM (PyTorch) | Prédiction énergie | R² | 0.9124 |
| LSTM (TF/Keras) | Régression carburant | MAE | ~0.80 L/100km |
| DistilBERT | Sentiment | Score | 90.15% |
| BERT-large | NER | Confiance max | 1.000 (Germany) |

---

*BARHOINE AYOUB — FST Settat — Deep Learning — 2026*
