from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch

MODEL_PATH = "./output"
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

id2label = model.config.id2label

app = FastAPI()

class TextRequest(BaseModel):
    text: str

@app.post("/predict")
def predict(req: TextRequest):
    """
    Predicts the category of a given text input using a fine-tuned BERT model.

    This endpoint receives a text string, processes it through a BERT tokenizer,
    performs inference with the classification model, and returns the predicted
    category label along with the confidence score.

    Args:
        req (TextRequest): A Pydantic request body containing the `text` to classify.

    Returns:
        dict: A dictionary with:
            - "label" (str): The predicted class label.
            - "confidence" (float): The confidence score of the prediction (between 0 and 1).
    """
    inputs = tokenizer(req.text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        predicted_class_id = torch.argmax(probs, dim=1).item()
        label = id2label[int(predicted_class_id)]
        return {
            "label": label,
            "confidence": round(probs[0][predicted_class_id].item(), 3)
        }

#Use the command line below to test in other terminal the working API in the container
#curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"text": "Hello Mr Cami"}'
