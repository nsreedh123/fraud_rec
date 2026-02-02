from trainer import FraudDataPreprocessor, load_dataset, plot_target_distribution

if __name__ == "__main__":
    # Load dataset
    train, train_id, test, test_id = load_dataset(data_path="dataset/")

    # Preprocess data
    preprocessor = FraudDataPreprocessor()
    train_processed, test_dataset = preprocessor.fit_transform(train, train_id, test, test_id)

    # Plot target distribution
    plot_target_distribution(train_processed)