import os
import pickle
import glob
import shutil
from sklearn.metrics import accuracy_score

# Import training function and XML loader from train.py
from train import train_model, XMLLoader

# Import prediction function from test.py
from test import predict_part

def run_cross_validation_experiment(
    data_folder='output_parts',
    saved_model_filename='absa_model.pkl'
):
    """
    This function performs 10-fold cross-validation using the ABSA pipeline.
    It trains on 9 parts and evaluates on the remaining part, repeating this for each fold.
    At the end, it reports accuracy for each fold and the average accuracy across all folds.
    """

    part_accuracies = []  # List to store accuracy for each part

    for test_part_index in range(1, 11):
        print(f"\nPart {test_part_index} ")

        # 1) Create a temporary directory to store training data (9 parts)
        temporary_training_dir = 'temp_training_parts'
        if os.path.exists(temporary_training_dir):
            shutil.rmtree(temporary_training_dir)
        os.makedirs(temporary_training_dir)

        # 2) Copy the 9 training XML files into the temporary directory
        for part_number in range(1, 11):
            if part_number == test_part_index:
                continue  # skip test part
            source_path = os.path.join(data_folder, f'part{part_number}.xml')
            destination_path = os.path.join(temporary_training_dir, f'part{part_number}.xml')
            shutil.copy(source_path, destination_path)

        # 3) Train the model on the current 9-fold training set
        train_model(temporary_training_dir, saved_model_filename)

        # 4) Load the test part (the 10th XML file)
        test_file_path = os.path.join(data_folder, f'part{test_part_index}.xml')
        test_texts, test_metadata, gold_labels = XMLLoader([test_file_path]).load()
        test_feature_input = list(zip(test_texts, test_metadata))

        # 5) Load the trained model from disk
        with open(saved_model_filename, 'rb') as file:
            trained_pipeline = pickle.load(file)

        # Optional: Print number of final features
        try:
            feature_union = trained_pipeline.named_steps['features']
            feature_matrix = feature_union.transform(test_feature_input)
            print(f"Number of features in test set: {feature_matrix.shape[1]}")
        except Exception as e:
            print(f"Could not compute feature count: {e}")

        # 6) Make predictions using the trained model
        predicted_labels = trained_pipeline.predict(test_feature_input)

        # 7) Compute accuracy for the current part
        part_accuracy = accuracy_score(gold_labels, predicted_labels)
        part_accuracies.append(part_accuracy)
        print(f"Accuracy for part {test_part_index}: {part_accuracy:.4f}")

    # 8) Print all part accuracies and the average
    print("\nCross-Validation Results ")
    for i, accuracy in enumerate(part_accuracies, start=1):
        print(f"Part {i}: {accuracy:.4f}")
    average_accuracy = sum(part_accuracies) / len(part_accuracies)
    print(f"Average Accuracy: {average_accuracy:.4f}")


if __name__ == "__main__":
    # Run the 10-part CV experiment
    run_cross_validation_experiment()


