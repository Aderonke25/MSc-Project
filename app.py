
import json
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Define the TransformerEncoder class (needed for ViT)
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = models.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False, mask=None):
        attn_output = self.att(inputs, inputs, attention_mask=mask, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    @classmethod
    def from_config(cls, config):
        config = {k: v for k, v in config.items() if k in ['embed_dim', 'num_heads', 'ff_dim', 'rate']}
        return cls(**config)

    def get_config(self):
        config = super(TransformerEncoder, self).get_config()
        config.update({
            'embed_dim': self.att.key_dim,
            'num_heads': self.att.num_heads,
            'ff_dim': self.ffn.layers[0].units,
            'rate': self.dropout1.rate,
        })
        return config

class PositionalEncoding(layers.Layer):
    def __init__(self, num_patches, embed_dim):
        super(PositionalEncoding, self).__init__()
        self.num_patches = num_patches
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=embed_dim)

    def call(self, patch_embeddings):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = patch_embeddings + self.position_embedding(positions)
        return encoded
    
    @classmethod
    def from_config(cls, config):
        return cls(num_patches=config['num_patches'], embed_dim=config['embed_dim'])

    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({
            'num_patches': self.num_patches,
            'embed_dim': self.position_embedding.output_dim,
        })
        return config


# Custom CNN model with attention mechanism
class CNNModel(tf.keras.Model):
    def __init__(self, num_classes, num_heads=4, attention_embed_dim=128, **kwargs):
        super(CNNModel, self).__init__(**kwargs)

        self.num_classes = num_classes
        self.num_heads = num_heads
        self.attention_embed_dim = attention_embed_dim

        # Convolutional Layers
        self.conv1 = layers.Conv2D(32, (3, 3), activation='relu', input_shape=(152, 152, 3))
        self.pool1 = layers.MaxPooling2D((2, 2))

        self.conv2 = layers.Conv2D(64, (3, 3), activation='relu')
        self.pool2 = layers.MaxPooling2D((2, 2))

        self.conv3 = layers.Conv2D(128, (3, 3), activation='relu')
        self.pool3 = layers.MaxPooling2D((2, 2))

        self.conv4 = layers.Conv2D(128, (3, 3), activation='relu')
        self.pool4 = layers.MaxPooling2D((2, 2))

        # Flatten layer
        self.flatten = layers.Flatten()

        # Fully Connected Dense Layer
        self.fc1 = layers.Dense(128, activation='relu')

        # Self-Attention Layer
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=attention_embed_dim)

        # Optional Dropout layer for regularization
        self.dropout = layers.Dropout(0.5)

        # Output layer
        self.fc2 = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        # Ensure inputs are in the correct shape (batch_size, height, width, channels)
        if len(inputs.shape) == 3:
            inputs = tf.expand_dims(inputs, axis=0)

        # Convolutional and Pooling layers
        x = self.conv1(inputs)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.pool4(x)

        # Flatten the convolution output
        x = self.flatten(x)

        # Dense layer
        x = self.fc1(x)

        # Reshape the flattened output to (batch_size, 1, feature_size) to feed into attention
        x_reshaped = tf.expand_dims(x, axis=1)

        # Apply self-attention
        x = self.attention(x_reshaped, x_reshaped)

        # Flatten the output from attention
        x = tf.squeeze(x, axis=1)

        # Dropout and final classification
        x = self.dropout(x)
        return self.fc2(x)


# GNN MODEL
class GraphConvLayer(layers.Layer):
    def __init__(self, units):
        super(GraphConvLayer, self).__init__()
        self.units = units
        self.dense = layers.Dense(units)

    def build(self, input_shape):
        self.dense.build(input_shape)
        super(GraphConvLayer, self).build(input_shape)

    def call(self, inputs, adjacency_matrix):
        x = inputs
        aggregated_neighbors = tf.matmul(adjacency_matrix, x)
        out = self.dense(aggregated_neighbors)
        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.units)

class GNNModel(tf.keras.Model):
    def __init__(self, num_classes, num_nodes, adjacency_matrix):
        super(GNNModel, self).__init__()
        self.num_nodes = num_nodes
        self.adjacency_matrix = adjacency_matrix

        # GNN Layers
        self.gnn1 = GraphConvLayer(128)
        self.gnn2 = GraphConvLayer(256)

        # Pooling and fully connected layers
        self.global_pool = layers.GlobalAveragePooling1D()
        self.fc1 = layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.dropout = layers.Dropout(0.5)
        self.fc2 = layers.Dense(num_classes, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.01))

    def call(self, inputs):
        # Ensure inputs are in the correct shape (batch_size, height, width, channels)
        if len(inputs.shape) == 3:
            inputs = tf.expand_dims(inputs, axis=0)

        # Extract patches from images
        patches = tf.image.extract_patches(
            images=inputs,
            sizes=[1, 38, 38, 1],
            strides=[1, 38, 38, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        batch_size = tf.shape(patches)[0]
        patch_size = 38 * 38 * 3
        patches = tf.reshape(patches, (batch_size, self.num_nodes, patch_size))

        # Apply GNN layers to the patches
        x = self.gnn1(patches, self.adjacency_matrix)
        x = tf.nn.relu(x)

        x = self.gnn2(x, self.adjacency_matrix)
        x = tf.nn.relu(x)

        # Pooling to reduce to a fixed-size vector
        x = self.global_pool(x)

        # Fully connected layers for classification
        x = self.fc1(x)
        x = self.dropout(x)
        return self.fc2(x)

# Define the adjacency matrix (connecting neighboring patches)
def create_adjacency_matrix(num_nodes):
    adjacency_matrix = tf.eye(num_nodes)
    return adjacency_matrix

# Function to load model based on user selection
def load_model_with_custom_objects(model_choice):
    if model_choice == 'ViT':
        return load_model('vit_model.h5', custom_objects={
            'TransformerEncoder': TransformerEncoder,
            'PositionalEncoding': PositionalEncoding
        })

    elif model_choice == 'CNN':
        model = CNNModel(num_classes=3, num_heads=4, attention_embed_dim=128)
        input_shape = (None, 152, 152, 3)
        model.build(input_shape)
        try:
            model.load_weights('cnn_model.h5')
        except Exception as e:
            print(f"Error loading CNN model weights: {str(e)}")
        return model

    elif model_choice == 'GNN':
        num_nodes = 16
        adjacency_matrix = create_adjacency_matrix(num_nodes)
        model = GNNModel(num_classes=3, num_nodes=num_nodes, adjacency_matrix=adjacency_matrix)
        input_shape = (None, 152, 152, 3)
        model.build(input_shape)
        try:
            model.load_weights('gnn_model.h5')
        except Exception as e:
            print(f"Error loading GNN model weights: {str(e)}")
        return model

    else:
        return None

# Extract class names
def get_class_names(model):
    try:
        with open('class_names.json', 'r') as f:
            actual_class_names = json.load(f)
        print('Class names loaded from file: ', actual_class_names)
        return actual_class_names
    except FileNotFoundError:
        print("Class names file not found, attempting to infer from model...")

# Streamlit App
logo = "https://image.pitchbook.com/2t5VYCiupSUlu28XjF6ggcmIQ061710842061323_200x200"
st.image(logo, width=120)

st.title("Image Classification App")

model_choice = st.selectbox("Select a model:", ('ViT', 'CNN', 'GNN'))

model = load_model_with_custom_objects(model_choice)
if model is not None:
    st.write(f"Model {model_choice} loaded successfully.")
else:
    st.error("Failed to load the selected model.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert grayscale images to RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')

    img = image.resize((152, 152))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image

    class_names = get_class_names(model)

    if st.button("Classify Image"):
        predictions = model.predict(img_array)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = class_names[predicted_class_idx]
        confidence = predictions[0][predicted_class_idx]

        if confidence < 0.5:
            st.error("Error: Image not known (Confidence below 50%)")
        else:
            st.write(f"Predicted Class: {predicted_class} (Confidence: {confidence:.2f})")
            st.write("Confidence Scores:")
            fig, ax = plt.subplots()
            ax.barh(class_names, predictions[0], color='blue')
            ax.set_xlim([0, 1])
            ax.set_xlabel('Confidence')
            ax.set_title('Class Confidence')
            plt.tight_layout()
            st.pyplot(fig)
