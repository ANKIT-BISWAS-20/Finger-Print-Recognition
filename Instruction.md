
## 📝 Setup Instructions

Follow the steps below to set up and run the project:

### 📦 1. Download the Dataset

Download the SOCOFing dataset from Kaggle:
🔗 [https://www.kaggle.com/datasets/ruizgara/socofing](https://www.kaggle.com/datasets/ruizgara/socofing)

Extract the dataset and place the `SOCOFing/` folder in the project root directory.

---

### 🐍 2. Create and Activate Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
venv\Scripts\activate
```

---

### 📥 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 🧠 4. Generate Descriptors (.pkl file)

```bash
python generate_descriptors.py
```

This will create a `real_fingerprints.pkl` file containing the fingerprint descriptors.

---

### 🚀 5. Run the Application

```bash
uvicorn main:app --reload
```

The application will start locally at:
👉 `http://127.0.0.1:8000`


