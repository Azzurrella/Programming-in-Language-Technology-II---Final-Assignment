import os
import pickle

# bring in exactly the names your pickled model expects under __main__:
from train import XMLLoader, POSFeatures, NERCount

# unpickle‐friendly helpers (already in your test.py)
def extract_text(X):
    """Unpickle helper: given [(text,meta),…], return [text,…]."""
    return [t for t,_ in X]

def extract_meta(X):
    """Unpickle helper: given [(text,meta),…], return [meta,…]."""
    return [m for _,m in X]

def predict_part(part_dir, part_number, model_path='absa_model.pkl'):
    """
    Load a trained ABSA pipeline and predict polarities for one partX.xml.
    Returns a list of dicts: {sentence, category, predicted_polarity}.
    """
    # 1) load pipeline
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # 2) load that one XML
    xml_file = os.path.join(part_dir, f'part{part_number}.xml')
    if not os.path.isfile(xml_file):
        raise FileNotFoundError(f"No such file: {xml_file!r}")
    X_text, X_meta, _ = XMLLoader([xml_file]).load()

    # 3) predict
    X = list(zip(X_text, X_meta))
    y_pred = model.predict(X)

    # 4) return structured results
    return [
        {
            'sentence': sentence,
            'category': meta['category'],
            'predicted_polarity': polarity
        }
        for (sentence, meta), polarity in zip(X, y_pred)
    ]


if __name__ == '__main__':
    # example: predict on part 2
    results = predict_part('./output_parts/', 2, './model')
    for r in results:
        print(r)
