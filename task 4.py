import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import itertools
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers, callbacks
import cv2

# ---------- Config ----------
DATA_DIR = "dataset"         # Folder with gesture images, e.g. dataset/class1, dataset/class2
IMG_SIZE = (160, 160)        # Input size for MobileNetV2
BATCH_SIZE = 32
EPOCHS = 20
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "gesture_mobilenetv2.h5")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")
# ----------------------------

print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

# Data generators: augmentation for training, only rescaling for validation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

num_classes = len(train_gen.class_indices)
print("Classes found:", train_gen.class_indices)

# Save label indices to JSON for later use
with open(LABELS_PATH, "w") as f:
    json.dump(train_gen.class_indices, f)

# Build the model with MobileNetV2 base
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs * 255.)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.25)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)

model = models.Model(inputs, outputs)
model.compile(optimizer=optimizers.Adam(1e-3),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Callbacks
cb_list = [
    callbacks.ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
    callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1)
]

# Train the head
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=cb_list
)

# Fine-tune some layers of base model
base_model.trainable = True
fine_tune_at = 100  # unfreeze from this layer to the end
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=optimizers.Adam(1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

ft_epochs = 10
total_epochs = EPOCHS + ft_epochs

history_fine = model.fit(
    train_gen,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1] + 1,
    validation_data=val_gen,
    callbacks=cb_list
)

# Save final model & labels again
model.save(MODEL_PATH)
with open(LABELS_PATH, "w") as f:
    json.dump(train_gen.class_indices, f)

print(f"Model saved to {MODEL_PATH}")
print(f"Labels saved to {LABELS_PATH}")

# Evaluation
val_steps = int(np.ceil(val_gen.samples / val_gen.batch_size))
y_pred_probs = model.predict(val_gen, steps=val_steps)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = val_gen.classes

idx_to_class = {v: k for k, v in train_gen.class_indices.items()}
target_names = [idx_to_class[i] for i in range(num_classes)]

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=target_names))

cm = confusion_matrix(y_true, y_pred)

def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(8,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i,j], 'd'),
                 horizontalalignment='center',
                 color='white' if cm[i,j] > thresh else 'black')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

plot_confusion_matrix(cm, target_names)

# Plot training curves
def plot_history(h1, h2=None):
    acc = h1.history.get('accuracy', [])
    val_acc = h1.history.get('val_accuracy', [])
    loss = h1.history.get('loss', [])
    val_loss = h1.history.get('val_loss', [])

    if h2:
        acc += h2.history.get('accuracy', [])
        val_acc += h2.history.get('val_accuracy', [])
        loss += h2.history.get('loss', [])
        val_loss += h2.history.get('val_loss', [])

    epochs = range(1, len(acc) + 1)
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(epochs, acc, label='train_acc')
    plt.plot(epochs, val_acc, label='val_acc')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Loss')
    plt.legend()
    plt.show()

plot_history(history, history_fine)

# --- Real-time webcam inference ---
def load_labels(path):
    with open(path, "r") as f:
        data = json.load(f)
    return {v:k for k,v in data.items()}

labels_map = load_labels(LABELS_PATH)

def predict_frame(frame, model, labels_map):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMG_SIZE)
    img_array = img_resized.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    class_idx = np.argmax(preds, axis=1)[0]
    prob = preds[0][class_idx]
    return labels_map[class_idx], prob

def webcam_inference(model, labels_map):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return
    print("Press 'q' to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        label, prob = predict_frame(frame, model, labels_map)
        cv2.putText(frame, f"{label} ({prob:.2f})", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)

        cv2.imshow("Hand Gesture Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Uncomment below to run live webcam inference after training
# webcam_inference(model, labels_map)
