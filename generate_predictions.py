import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import Subset
from torch.optim.lr_scheduler import StepLR
from PIL import Image
import os

def generate_prediction(img_data_path):
    map = {}
    map[0] = 'aloevera'
    map[1] = 'banana'
    map[2] = 'bilimbi'
    map[3] = 'cantaloupe'
    map[4] = 'cassava'
    map[5] = 'coconut'
    map[6] = 'corn'
    map[7] = 'cucumber'
    map[8] = 'curcuma'
    map[9] = 'eggplant'
    map[10] = 'galangal'
    map[11] = 'ginger'
    map[12] = 'guava'
    map[13] = 'kale'
    map[14] = 'longbeans'
    map[15] = 'mango'
    map[16] = 'melon'
    map[17] = 'orange'
    map[18] = 'paddy'
    map[19] = 'papaya'
    map[20] = 'peper chili'
    map[21] = 'pineapple'
    map[22] = 'pomelo'
    map[23] = 'shallot'
    map[24] = 'soybeans'
    map[25] = 'spinach'
    map[26] = 'sweet potatoes'
    map[27] = 'tobacco'
    map[28] = 'waterapple'
    map[29] = 'watermelon'

    # Define the transformation to apply to the input images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the input image to 224x224
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
    ])

    model_path = 'best_model.8280.pt'

    device = torch.device("mps" if torch.cuda.is_available() else "cpu")
    # Load the model
    model = torch.load(model_path, map_location=torch.device('cpu'))

    image_path = img_data_path
    image = Image.open(image_path)


    # Apply the transformations
    input_tensor = transform(image)
    input_tensor = input_tensor.unsqueeze(0)  # Add a batch dimension

    # Move the input tensor to the device
    input_tensor = input_tensor.to(device)

    # Forward pass through the model
    with torch.no_grad():
        outputs = model(input_tensor)

    # Get the predicted class index
    _, predicted_idx = torch.max(outputs, 1)

    # Convert the tensor to a numerical value
    predicted_class = predicted_idx.item()

    # Print the predicted class
    print('Predicted class:', map[predicted_class])

def main():
    img_file = 'dataset_type_of_plants_new/kale/kale12.jpg'
    generate_prediction(img_file)

if __name__ == '__main__':
    main()