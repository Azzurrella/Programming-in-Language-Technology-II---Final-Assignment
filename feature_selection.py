import os
import shutil
import glob
from train import XMLLoader, POSFeatures, NERCount, extract_text, extract_meta
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score


def build_pipeline_with_feature_selection(k_best_features):
    """
    Constructs a machine learning pipeline that extracts multiple types of features,
    applies feature selection using chi-squared test, and finally trains a Linear SVM classifier.
    """

    # Pipeline for extracting TF-IDF features from raw review text
    tfidf_pipeline = Pipeline([
        ('extract_text', FunctionTransformer(extract_text, validate=False)),
        ('tfidf_vectorizer', TfidfVectorizer(ngram_range=(1, 3), min_df=2)),
    ])

    # Pipeline for extracting POS n-grams from review text
    pos_pipeline = Pipeline([
        ('extract_text', FunctionTransformer(extract_text, validate=False)),
        ('pos_tagger', POSFeatures()),
        ('count_vectorizer', CountVectorizer(ngram_range=(1, 2))),
    ])

    # Pipeline for extracting metadata-based categorical features (e.g., polarity)
    metadata_pipeline = Pipeline([
        ('extract_meta', FunctionTransformer(extract_meta, validate=False)),
        ('dict_vectorizer', DictVectorizer()),
    ])

    # Pipeline for extracting Named Entity Recognition counts
    ner_pipeline = Pipeline([
        ('extract_text', FunctionTransformer(extract_text, validate=False)),
        ('ner_counter', NERCount()),
        ('dict_vectorizer', DictVectorizer()),
    ])

    # Combine all feature extractors into a single unified feature space
    combined_features = FeatureUnion([
        ('tfidf_features', tfidf_pipeline),
        ('pos_features', pos_pipeline),
        ('metadata_features', metadata_pipeline),
        ('ner_features', ner_pipeline),
    ])

    # Full ML pipeline: feature extraction → feature selection → classification
    pipeline = Pipeline([
        ('feature_union', combined_features),
        ('feature_selection', SelectKBest(chi2, k=k_best_features)),
        ('svm_classifier', LinearSVC(C=1.0, max_iter=20000)),
    ])
    return pipeline


def run_cross_validation_with_feature_selection(k_best_features, input_dir='output_parts', model_save_path='absa_model.pkl'):
    """
    Performs 10-fold cross-validation on XML data parts.
    Each fold is evaluated using a pipeline that applies chi2 feature selection.
    The top TF-IDF features are printed to help understand which words are most predictive.
    """

    print(f"\nFeature Selection with k = {k_best_features} ")
    part_accuracies = []

    for test_part_index in range(1, 11):
        print(f"\nPart {test_part_index} ")

        # 1) Prepare temporary directory with 9 folds for training
        training_dir = 'temp_training_data'
        if os.path.exists(training_dir):
            shutil.rmtree(training_dir)
        os.makedirs(training_dir)

        # Copy 9 parts for training (leaving one part for testing)
        for i in range(1, 11):
            if i != test_part_index:
                shutil.copy(
                    os.path.join(input_dir, f'part{i}.xml'),
                    os.path.join(training_dir, f'part{i}.xml')
                )

        # 2) Load training data from XML files
        train_file_paths = glob.glob(os.path.join(training_dir, 'part*.xml'))
        train_texts, train_metadata, train_labels = XMLLoader(train_file_paths).load()
        training_instances = list(zip(train_texts, train_metadata))

        # 3) Load test data from the held-out part
        test_file = os.path.join(input_dir, f'part{test_part_index}.xml')
        test_texts, test_metadata, test_labels = XMLLoader([test_file]).load()
        test_instances = list(zip(test_texts, test_metadata))

        # 4) Build pipeline and fit model on training data
        pipeline = build_pipeline_with_feature_selection(k_best_features)
        print(f"Training on {len(training_instances)} instances…")
        pipeline.fit(training_instances, train_labels)

        # 5) Make predictions and compute accuracy
        predictions = pipeline.predict(test_instances)
        accuracy = accuracy_score(test_labels, predictions)
        part_accuracies.append(accuracy)

        # 6) Report selected feature count and top TF-IDF features
        try:
            selected_indices = pipeline.named_steps['feature_selection'].get_support(indices=True)
            print(f"Number of features after selection: {len(selected_indices)}")

            # Get top TF-IDF feature names
            tfidf_step = pipeline.named_steps['feature_union'].transformer_list[0][1]
            tfidf_vectorizer = tfidf_step.named_steps['tfidf_vectorizer']
            tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()

            total_tfidf_features = tfidf_vectorizer.transform([training_instances[0][0]]).shape[1]
            selected_tfidf_indices = [i for i in selected_indices if i < total_tfidf_features]
            selected_tfidf_names = tfidf_feature_names[selected_tfidf_indices]

            # Print top 20 TF-IDF features
            print("\nTop 20 selected TF-IDF features:")
            for idx, feature_name in enumerate(selected_tfidf_names[:20]):
                print(f"{idx + 1}. {feature_name}")

        except Exception as e:
            print(f"Could not extract TF-IDF feature names: {e}")

    # 7) Print average accuracy across all parts
    average_accuracy = sum(part_accuracies) / len(part_accuracies)
    print(f"\nResults for k = {k_best_features} ")
    for idx, acc in enumerate(part_accuracies, start=1):
        print(f"Part {idx}: {acc:.4f}")
    print(f"Average Accuracy: {average_accuracy:.4f}")


if __name__ == '__main__':
    # Run the experiment for two different values of k (number of selected features)
    run_cross_validation_with_feature_selection(k_best_features=2000)
    run_cross_validation_with_feature_selection(k_best_features=5000)
