import pickle

import tensorflow_datasets as tfds

(clinc_train, clinc_test, clinc_test_oos), ds_info = tfds.load(
    'clinc_oos', split=['train', 'test', 'test_oos'], with_info=True, batch_size=-1)


pickle.dump(clinc_train['text'].numpy(), open("data/oos_train_texts.pkl", "wb"))
pickle.dump(clinc_train['intent'].numpy(), open("data/oos_train_intents.pkl", "wb"))


pickle.dump(clinc_test['text'].numpy(), open("data/oos_test_texts.pkl", "wb"))
pickle.dump(clinc_test['intent'].numpy(), open("data/oos_test_intents.pkl", "wb"))


pickle.dump(clinc_test_oos['text'].numpy(), open("data/oos_oos_texts.pkl", "wb"))
pickle.dump(clinc_test_oos['intent'].numpy(), open("data/oos_oos_intents.pkl", "wb"))


# path = "data/oos_train_texts.pkl"
# arr = pickle.load(open(path, "rb"))
# arr = [a.decode("utf-8") for a in arr]
# pickle.dump(arr, open(path, "wb"))


# oos_text_test = list(clinc_test['text'].numpy()) +  list(clinc_test_oos['text'].numpy())
# oos_text_test = [text.decode("utf-8") for text in oos_text_test]

# pickle.dump(oos_text_test, open("oos_texts.pkl", "wb"))


# oos_intents = list(clinc_test['intent'].numpy()) +  list(clinc_test_oos['intent'].numpy())
# oos_intents = [text.decode("utf-8") for text in oos_texts]

# # oos_texts = pickle.load(open("oos_texts.pkl", "rb"))
