
# 🧠 Quora Question Pairs - Semantic Similarity Detection

**Client Project ID:** CLIENT-PROJECT-QQP  
**Dataset:** [Kaggle Quora Question Pairs](https://www.kaggle.com/competitions/quora-question-pairs)

---

## 📌 Business Case

On community Q&A platforms like Quora, users often post the same question in different ways. Detecting such semantically duplicate questions helps improve search results, reduce clutter, and enhance the overall user experience. This project builds a machine learning model that can automatically identify whether a pair of questions mean the same thing.

> **Note:** The labels in the dataset are human-generated and may include noise due to subjectivity in interpreting sentence meaning.

---

## 🎯 Project Goal

To build and train a model that accurately predicts whether a given pair of questions are duplicates (i.e., semantically similar).

---

## 📁 Dataset Features

| Feature       | Description                                            |
|---------------|--------------------------------------------------------|
| `id`          | Unique ID for the question pair                        |
| `qid1`, `qid2`| Unique IDs of each individual question                 |
| `question1`   | First question (text)                                  |
| `question2`   | Second question (text)                                 |
| `is_duplicate`| Target label – 1 if questions are duplicates, else 0   |

---

## 🧰 Tools & Technologies

- Python  
- Google Colab  
- Pandas, NumPy  
- NLTK (for text cleaning)  
- TensorFlow / Keras  
- Matplotlib & Seaborn  
- Scikit-learn  

---

## 🧠 Model Architecture

- **Preprocessing:** Cleaning text (lowercasing, removing stopwords, punctuation)
- **Tokenization:** Keras Tokenizer
- **Embedding:** Keras Embedding Layer
- **Model:** Siamese-style BiLSTM model using shared LSTM layers
- **Loss Function:** Binary Crossentropy
- **Optimizer:** Adam

---

## 📊 Performance

- **Accuracy:** ~80%  
- **Epochs Trained:** 5  
- **Early Stopping:** Used to avoid overfitting  
- **Visualization:** Accuracy and confusion matrix plotted

---

## 📈 Results

- The model is able to generalize well for detecting duplicate questions.
- Preprocessing and shared LSTM improved learning between paired text.
- Can be integrated into a Quora-like system to detect duplicates in real time.

---

## 🧪 How to Use

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/Quora-Question-Pairs-Client-Project.git
   cd Quora-Question-Pairs-Client-Project
   ```

2. Upload the dataset (`train.csv`) from Kaggle.

3. Run the notebook or script:
   - `quora_question_similarity_model.py`

4. The trained model will be saved as:
   ```
   quora_question_similarity_model.h5
   ```

---

## 💡 Future Enhancements

- Use transformer-based models (like BERT or DistilBERT)
- Add attention layers for improved representation
- Deploy as a web service using Flask or Streamlit
- Use pre-trained sentence embeddings (SBERT, USE, etc.)

---

## 👤 Author

**[Sasidhara Srivatchasa]**  
Intern, AI & Machine Learning Client Project – February 2025  

---

## 📄 License

This project is for academic and internship learning purposes only.

```
