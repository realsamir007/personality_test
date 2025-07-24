# 🧠 Introvert vs Extrovert Personality Predictor

This is a simple and interactive Streamlit app that predicts whether a person is an **Introvert** or an **Extrovert** based on their social behavior, using a trained **LightGBM** model exported in **ONNX** format.

---

## 🚀 Features

- Interactive UI built with **Streamlit**
- Lightweight and fast **LightGBM model**
- Exported to **ONNX** for efficient deployment
- Takes 7 input features related to social behavior
- Real-time prediction of personality type

---

## 📊 Dataset

The model is trained on a custom dataset `intro_extro.csv` with the following features:

| Feature                    | Description                                           |
|----------------------------|-------------------------------------------------------|
| `Time_spent_Alone`         | Time (in hours) spent alone per day                  |
| `Stage_fear`               | Whether the person fears public speaking             |
| `Social_event_attendance`  | Frequency of attending social events (0–10 scale)    |
| `Going_outside`            | Daily hours spent outside                            |
| `Drained_after_socializing` | Feels drained after social interactions (Yes/No)    |
| `Friends_circle_size`      | Approximate number of close friends                  |
| `Post_frequency`           | Frequency of posting on social media (0–10 scale)    |

The target variable is `Personality` with:
- `0 = Introvert`
- `1 = Extrovert`

---

## ⚙️ Installation & Running the App

### 🔧 1. Install dependencies

```bash
pip install -r requirements.txt

