#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf

import numpy as np
import os
import time
import chardet

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)


# In[2]:


texts = []
dictionary = set()
text = ''

for i in range(15):
    for j in range(49):
        path_to_file = '/home/jupyter/data/g' + str(i) + '/d'+str(j)+'.txt'
            
        file = open(path_to_file, "rb")
        rawdata = file.read()
        file.close()
        result = chardet.detect(rawdata)
        
        try:
            decoded=rawdata.decode(encoding=result['encoding'])
        except:
            None
        else:
            if result['encoding'] == "utf-8":
                formated = decoded.strip()
                formated = formated.replace('\r', '')
                #texts.append(formated)
                text += formated
                dictionary = dictionary.union(set(formated))


# In[3]:


vocab = sorted(dictionary)
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)


# In[4]:


#int_texts = []
#for txt in texts:
#    int_texts.append(np.array([char2idx[c] for c in txt]))
int_text = np.array([char2idx[c] for c in text])

print("Vocabulary elements", len(vocab))
print("Documents #", len(texts))
print("Text characters #", len(text))


# In[5]:


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


# In[6]:


def get_sequences(txt_int):
    # Create training examples / targets
    seq_length = 80
    char_dataset = tf.data.Dataset.from_tensor_slices(txt_int)
    sequences = char_dataset.batch(seq_length+1, drop_remainder=True)
    return sequences.map(split_input_target)


# In[7]:


# ============================================== Checkpoint ==============================================
# ============================================== Checkpoint ==============================================
# The maximum length sentence
# datasets = []

dataset = get_sequences(int_text)

#for int_text in int_texts:
#    datasets.append(get_sequences(int_text))


# In[8]:


# for input_example, target_example in  datasets[0].take(1):
for input_example, target_example in  dataset.take(1):    
    print ('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
    print ('Target data:', repr(''.join(idx2char[target_example.numpy()])))


# In[9]:


# Batch size
BATCH_SIZE = 128

# Buffer size to shuffle the dataset. Maintains a buffer in which it shuffles elements), not all dataset.
BUFFER_SIZE = 10000
#BUFFER_SIZE = 500

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

#for i in range(len(datasets)):
#    datasets[i] = datasets[i].shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)


# In[10]:


# Length of the vocabulary in chars
vocab_size = len(vocab)

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1024


# In[11]:


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model


# In[12]:


model = build_model(
  vocab_size = vocab_size,
  embedding_dim=embedding_dim,
  rnn_units=rnn_units,
  batch_size=BATCH_SIZE)


# In[13]:


optimizer = tf.keras.optimizers.Adam()


# In[14]:


@tf.function
def train_step(inp, target):
    with tf.GradientTape() as tape:
        predictions = model(inp)
        loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                target, predictions, from_logits=True))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss


# In[ ]:


# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints4'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

# Training step
EPOCHS = 12

for epoch in range(EPOCHS):
    start = time.time()

    # initializing the hidden state at the start of every epoch
    # initally hidden is None
    hidden = model.reset_states()

    #for (batch_n, (inp, target)) in enumerate(datasets[0]):
    for (batch_n, (inp, target)) in enumerate(dataset):
        loss = train_step(inp, target)
        
        if batch_n % 200 == 0:
            template = 'Epoch {} Batch {} Loss {}'
            print(template.format(epoch+1, batch_n, loss))

    # saving (checkpoint) the model every 5 epochs
    if (epoch + 1) % 2 == 0:
        model.save_weights(checkpoint_prefix.format(epoch=epoch))

    print ('Epoch {} Loss {:.4f}'.format(epoch+1, loss))
    print ('Time taken for epoch {} sec\n'.format(time.time() - start))

print("Done")
model.save_weights(checkpoint_prefix.format(epoch=epoch))


# In[ ]:


print(tf.train.latest_checkpoint(checkpoint_dir))

#'./training_checkpoints2/ckpt_6'
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))


# In[ ]:


model.summary()


# In[ ]:


def generate_text(model, start_string):
    # Evaluation step (generating text using the learned model)

    # Number of characters to generate
    num_generate = 500

    # Converting our start string to numbers (vectorizing)
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our results
    text_generated = []
    temperature = 0.9

    # Here batch size == 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # using a categorical distribution to predict the character returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
        
        # We pass the predicted character as the next input to the model
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))


# In[ ]:


print("Creating")
print(generate_text(model, start_string=u"<html"))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




