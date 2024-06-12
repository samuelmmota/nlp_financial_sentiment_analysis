import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import os
import json

INPUT_DATA_FOLDER = 'src/tempdata/articles'
OUTPUT_JSON_FILE = 'dataset/output_data_sentiment.json'


def prepare_dataset(input_data_folder: str = INPUT_DATA_FOLDER, output_json_file: str = OUTPUT_JSON_FILE):
    tokenizer = AutoTokenizer.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
    model = AutoModelForSequenceClassification.from_pretrained(
        "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")

    all_items = []
    json_files = [f for f in os.listdir(input_data_folder) if f.endswith('.json')]

    for json_file in json_files:
        file_path = os.path.join(input_data_folder, json_file)
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

            if isinstance(data, list):
                for item in data:
                    if 'fields' in item:
                        text = item['fields']['bodyText']
                        print("Predicting Class and Label for text:", text)
                        # split the input text into smaller segments
                        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True,
                                           return_overflowing_tokens=True)

                        all_logits = []
                        for input_ids in inputs['input_ids']:
                            output = model(input_ids.unsqueeze(0))
                            logits = output.logits
                            all_logits.append(logits)

                        avg_logits = torch.mean(torch.stack(all_logits), dim=0)
                        print("Logits:", avg_logits)

                        probabilities = F.softmax(avg_logits, dim=-1)
                        print("Probabilities:", probabilities)

                        predicted_class_id = torch.argmax(probabilities, dim=-1).item()
                        predicted_label = model.config.id2label[predicted_class_id]

                        print("Predicted class ID:", predicted_class_id)
                        print("Predicted Label:", predicted_label)

                        item['fields']['sentiment'] = predicted_class_id
                        item['fields']['sentimentLabel'] = predicted_label
                all_items.extend(data)
            else:
                print(f"Warning: {file_path} does not contain a list of items.")

    with open(output_json_file, 'w', encoding='utf-8') as file:
        json.dump(all_items, file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # Test
    prepare_dataset()
