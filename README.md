# krishi-mitra-AI
🌾 Agriculture Yield Advisor – A Streamlit app that predicts crop yield using ML and explains results with Generative AI. Get improvement tips, relevant government schemes, and ask follow-up questions in an interactive chatbot.

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token_optional
```

`HF_TOKEN` is optional if the model repo is public.

3. Run the app:

```bash
streamlit run app.py
```

## Compatibility Notes

- The hosted model was serialized with scikit-learn 1.6.1 internals.
- The app includes a compatibility patch for newer scikit-learn versions.
- If model loading still fails, use Python 3.12 and install `scikit-learn==1.6.1`.

