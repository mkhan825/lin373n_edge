import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import tensorflow_datasets as tfds

tags = set()
# if is_training:
tags.add("train")
albert_module = hub.Module("https://tfhub.dev/google/albert_base/1", tags=tags,
                           trainable=True)
albert_inputs = dict(
    input_ids=input_ids,
    input_mask=input_mask,
    segment_ids=segment_ids)
albert_outputs = albert_module(
    inputs=albert_inputs,
    signature="tokens",
    as_dict=True)

# If you want to use the token-level output, use
# albert_outputs["sequence_output"] instead.
output_layer = albert_outputs["pooled_output"]

albert_url ='https://tfhub.dev/tensorflow/albert_en_base/2'
encoder = hub.KerasLayer(albert_url)

preprocessor_url = "https://tfhub.dev/tensorflow/albert_en_preprocess/3"
preprocessor = hub.KerasLayer(preprocessor_url)

train_data, validation_data, test_data = tfds.load(
    name="imdb_reviews", 
    split=('train[:60%]', 'train[60%:]', 'test'),
    as_supervised=True)

text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
encoder_inputs = preprocessor(text_input)

outputs = encoder(encoder_inputs)
pooled_output = outputs["pooled_output"]     
embedding_model = tf.keras.Model(text_input, pooled_output)

model = tf.keras.Sequential()
model.add(embedding_model)
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(30, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(8, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(1))

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
history = model.fit(train_data.shuffle(10000).batch(128),
                    epochs=10,
                    validation_data=validation_data.batch(128),
                    verbose=1)
