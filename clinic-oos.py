import pickle

import tensorflow_datasets as tfds

(clinc_train, clinc_test, clinc_test_oos), ds_info = tfds.load(
    'clinc_oos', split=['train', 'test', 'test_oos'], with_info=True, batch_size=-1)

oos_texts = list(clinc_test['text'].numpy()) +  list(clinc_test_oos['text'].numpy())
oos_texts = [text.decode("utf-8") for text in oos_texts]

pickle.dump(oos_texts, open("oos_texts.pkl", "wb"))

# oos_texts = pickle.load(open("oos_texts.pkl", "rb"))
