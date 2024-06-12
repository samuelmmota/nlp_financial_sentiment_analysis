from src.prepare_dataset import prepare_dataset
from src.visualize import create_dataframe, mean_graph_plot, frequency_bar_plot

INPUT_JSON_DATASET = "dataset/output_data_sentiment.json"

if __name__ == "__main__":
    #prepare_dataset()
    df = create_dataframe(input_path=INPUT_JSON_DATASET)

    mean_graph_plot(df, export=True)
    frequency_bar_plot(df, export=True)
