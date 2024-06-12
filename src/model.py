import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F


def create_roberte_model():
    tokenizer = AutoTokenizer.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
    model = AutoModelForSequenceClassification.from_pretrained(
        "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")

    return tokenizer, model


def get_prediction_class_orig(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits
    print("Logits:", logits)
    probabilities = F.softmax(logits, dim=-1)
    print("Probabilities:", probabilities)

    predicted_class = torch.argmax(probabilities, dim=-1).item()
    predicted_class_id = logits.argmax().item()
    predicted_label = model.config.id2label[predicted_class_id]

    print("Predicted class:", predicted_class)
    print("Predicted Label:", predicted_label)

    return predicted_class


def get_prediction_class(model, tokenizer, text):
    print("Predicting Class and Label for text:", text)
    # split the input text into smaller segments
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True,
                       return_overflowing_tokens=True)

    all_logits = []
    for input_ids in inputs['input_ids']:
        output = model(input_ids.unsqueeze(0))
        logits = output.logits
        all_logits.append(logits)

    # Aggregate logits if needed (e.g., by averaging)
    avg_logits = torch.mean(torch.stack(all_logits), dim=0)
    print("Logits:", avg_logits)

    probabilities = F.softmax(avg_logits, dim=-1)
    print("Probabilities:", probabilities)

    predicted_class_id = torch.argmax(probabilities, dim=-1).item()
    predicted_label = model.config.id2label[predicted_class_id]

    print("Predicted class ID:", predicted_class_id)
    print("Predicted Label:", predicted_label)

    return predicted_class_id, predicted_label


if __name__ == "__main__":
    # TEST
    tokenizer, model = create_roberte_model()
    text = "Even after 200 years, Lord Byron's maiden speech in the House of Lords has a searing effect. Delivered two centuries ago this week, in response to plans to make the breaking of weaving machines a capital crime, it sprang from Byron's direct knowledge of the unemployed weavers' revolt in Nottinghamshire. \"Nothing but absolute want could have driven a large and once honest and industrious body of the people into the commission of excesses so hazardous to themselves, their families and their community,\" he began. \"You may call the people a mob, but do not forget that a mob too often speaks the sentiments of the people,\" he went on. \"Can you carry this bill into effect? Can you commit a whole county to their own prisons? Will you erect a gibbet in every field and hang up men like scarecrows?\" Two weeks later, Childe Harold was published and Byron awoke to find himself famous. But it is not just his poems that still ring down the ages."
    get_prediction_class(model, tokenizer, text)
