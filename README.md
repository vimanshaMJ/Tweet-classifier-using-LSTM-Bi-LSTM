# Tweet-classifier-using-LSTM-Bi-LSTM

A deep learning project that classifies tweets as personal health mentions (PHM) or not, using advanced LSTM and Bi-LSTM models.

---

## 🚀 Key Features

- **Robust NLP Pipeline:**  
  Text cleaning, tokenization, and sequence modeling for effective tweet processing.
- **Dual Model Approach:**  
  Implements both LSTM and Bi-LSTM architectures for comparative analysis.
- **High Accuracy:**  
  Achieves over 80% accuracy on real-world tweet data.
- **Performance Visualization:**  
  Includes accuracy/loss plots and confusion matrices.

---

## 🖥️ Interactive Interface

- **Real-time Tweet Classification:**  
  User-friendly web interface for instant predictions.
- **Model Comparison:**  
  View predictions from both models side-by-side.

---

## 📁 Project Structure

```
├── config/                
├── data/                  
│   ├── phm_train.csv
│   └── phm_test.csv
├── models/                
│   ├── bilstm_model.h5
│   ├── lstm_model.h5
│   └── tokenizer.pkl
├── notebooks/            
│   └── project.ipynb
├── src/                  
│   ├── interface.py
│   ├── predict.py
│   ├── train.py
│   └── utils.py
├── ProjectReport.pdf      
```

---

## 🚀 Launch Web Interface

python src/interface.py


## 🛠️ Technologies Used

- **Python**
- **Jupyter Notebook**
- **Pandas - Data manipulation and preprocessing, including cleaning, transforming, and organizing tweet data**
- **TensorFlow Keras - Building, training, and evaluating LSTM and Bi-LSTM deep learning models**
- **Matplotlib - Visualizing model performance, such as accuracy/loss plots**
- **scikit-learn (sklearn) - Data preprocessing, model evaluation, and providing metrics like confusion matrix and accuracy score**

