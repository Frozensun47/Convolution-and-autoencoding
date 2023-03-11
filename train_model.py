from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the saved model
model = load_model('my_model.h5')

# Define the directories containing the input images
input_dir = 'Images'

# Create data generators to load and preprocess the images
datagen = ImageDataGenerator()
batch_size = 8
generator = datagen.flow_from_directory(
    input_dir,
    target_size=(256, 256),
    batch_size=batch_size,
    class_mode='input',
    shuffle=True
)
steps_per_epoch = generator.n // batch_size
print("length of input data", len(generator))
num_samples = generator.samples
num_classes = generator.num_classes
input_shape = generator.image_shape
print(num_classes, num_samples, input_shape)


# Train the model using the training data
history = model.fit(generator, epochs=1, verbose=2)

# Save the updated model weights and architecture
model.save('my_updated_model.h5')

