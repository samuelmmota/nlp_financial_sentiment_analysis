import json
import pandas as pd
import matplotlib.pyplot as plt


def create_dataframe(input_path: str) -> pd.DataFrame:
    with open(input_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    extracted_data = []
    for item in data:
        if 'fields' in item and 'sentiment' in item['fields']:
            extracted_data.append({
                'id': item.get('id'),
                'webPublicationDate': item.get('webPublicationDate'),
                'sentiment': item['fields'].get('sentiment'),
                'sentimentLabel': item['fields'].get('sentimentLabel')
            })

    df = pd.DataFrame(extracted_data)

    df['webPublicationDate'] = pd.to_datetime(df['webPublicationDate'])
    df['year_month'] = df['webPublicationDate'].dt.to_period('M')
    return df


def mean_graph_plot(df: pd.DataFrame, export: bool = False):
    print("# Mean Sentiment trend analysis over the months")
    monthly_sentiment = df.groupby('year_month')['sentiment'].mean().reset_index()

    print(monthly_sentiment)
    monthly_sentiment['year_month'] = monthly_sentiment['year_month'].dt.to_timestamp()

    plt.figure(figsize=(10, 5))
    plt.plot(monthly_sentiment['year_month'], monthly_sentiment['sentiment'], marker='o')
    plt.title('Monthly Sentiment Trend')
    plt.xlabel('Month')
    plt.ylabel('Average Sentiment Score')
    plt.grid(True)
    plt.show()

    if export:
        monthly_sentiment.to_csv('exports/monthly_sentiment.csv', index=False)


def frequency_bar_plot(df: pd.DataFrame, export: bool = False):
    print("# Frequency sentiment class trend analysis over the months")
    sentiment_counts = df.groupby(['year_month', 'sentiment']).size().unstack(fill_value=0)

    print(sentiment_counts)
    sentiment_counts.index = sentiment_counts.index.to_timestamp()

    sentiment_counts.plot(kind='bar', stacked=True, figsize=(10, 5))
    plt.title('Monthly Sentiment Class Frequency')
    plt.xlabel('Month')
    plt.ylabel('Frequency')
    plt.legend(['Negative', 'Neutral', 'Positive'])
    plt.show()

    if export:
        sentiment_counts.to_csv('exports/monthly_sentiment_class_frequency.csv', index=True)


if __name__ == "__main__":
    # Test
    input_json_dataset = "../dataset/output_data_sentiment.json"
    df = create_dataframe(input_json_dataset)
    mean_graph_plot(df)
    frequency_bar_plot(df)
