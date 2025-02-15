import os
import torch
from torchvision.utils import save_image
from torchvision.transforms import ToTensor, Resize, CenterCrop
from PIL import Image
from torchvision.models import vgg19_bn

# Install dependencies
try:
    from pytorch_pretrained_biggan import BigGAN, one_hot_from_int, truncated_noise_sample
except ImportError:
    raise ImportError("Install pytorch_pretrained_biggan using `pip install pytorch-pretrained-biggan`")

# Step 1: Define output directory
output_dir = "generated_dataset"
os.makedirs(output_dir, exist_ok=True)

# Step 2: Load a pre-trained GAN model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BigGAN.from_pretrained('biggan-deep-256').to(device)
model.eval()

# Step 3: Preprocess the single input image
def preprocess_image(image_path, target_size=256):
    image = Image.open(image_path).convert('RGB')
    transform = CenterCrop((min(image.size), min(image.size)))
    image = transform(image)
    image = Resize((target_size, target_size))(image)
    image = ToTensor()(image)
    return image.unsqueeze(0).to(device)

# Path to your single input image
input_image_path = "path_to_your_image.jpg"
input_image = preprocess_image(input_image_path)

# Step 4: Generate multiple variations
num_samples = 10  # Number of augmented images to generate
truncation = 0.5  # Controls diversity (lower = less diverse)

for i in range(num_samples):
    # Generate random noise and class vector
    noise = truncated_noise_sample(truncation=truncation, batch_size=1)
    class_vector = one_hot_from_int([207], batch_size=1)  # Change class index as needed (207 is "golden retriever" for example)

    noise = torch.tensor(noise, dtype=torch.float32).to(device)
    class_vector = torch.tensor(class_vector, dtype=torch.float32).to(device)

    with torch.no_grad():
        output = model(noise, class_vector, truncation)

    # Save the generated image
    output_image_path = os.path.join(output_dir, f"generated_image_{i + 1}.png")
    save_image(output, output_image_path)

print(f"Generated {num_samples} augmented images in {output_dir}.")
