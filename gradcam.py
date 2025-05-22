# # Import necessary packages and libraries
# import torchvision
# import torch
# import numpy as np
# import torch.nn as nn
# from PIL import Image
# import torchvision.transforms as transforms
# import cv2
# import matplotlib.pyplot as plt


# # Load pre-trained model
# vgg_model = torchvision.models.vgg16(pretrained=True)

# # transformation for passing image into the network
# transform = transforms.Compose([
#     transforms.Resize((1080,1080)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.1080, 0.225])
# ])

# # selecting layers from the model to generate activations
# image_to_heatmaps = nn.Sequential(*list(vgg_model.features[:-4]))

# def compute_heatmap(model,img):
#   model.eval()
#   # compute logits from the model
#   logits = model(img)
#   # model's prediction 
#   pred = logits.max(-1)[-1]
#   # activations from the model
#   activations = image_to_heatmaps(img)
#   # compute gradients with respect to the model's most confident prediction
#   logits[0, pred].backward(retain_graph=True)
#   # average gradients of the featuremap 
#   pool_grads = model.features[-3].weight.grad.data.mean((0,2,3))
#   # multiply each activation map with corresponding gradient average
#   for i in range(activations.shape[1]):
#     activations[:,i,:,:] *= pool_grads[i]
#   # calculate mean of weighted activations
#   heatmap = torch.mean(activations, dim=1)[0].cpu().detach()
#   return heatmap, pred




# def upsampleHeatmap(map, image):
#   # permute image
#   image = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
#   # maximum and minimum value from heatmap
#   m, M = map.min(), map.max()
#   # normalize the heatmap
#   map = 255 * ((map-m)/ (m-M))
#   map = np.uint8(map)
#   # resize the heatmap to the same as the input
#   map = cv2.resize(map, (1080, 1080))
#   map = cv2.applyColorMap(255-map, cv2.COLORMAP_JET)
#   map = np.uint8(map)
#   # change this to balance between heatmap and image
#   map = np.uint8(map*0.7 + image*0.3)
#   return map


# def display_images(upsampled_map, image):
#     image = image.squeeze(0).permute(1, 2, 0)
#     fig, axes = plt.subplots(1, 2, figsize=(10, 5))

#     axes[0].imshow(upsampled_map)
#     axes[0].set_title("Heatmap")
#     axes[0].axis('off')
#     axes[1].imshow(image)
#     axes[1].set_title("Original Image")
#     axes[1].axis('off')
#     plt.show()




# # Example usage
# # cat_dog_img = "cat_and_dog.jpg"
# cat_dog_img = "./ISTD_Dataset/test/test_A/100-3.png"
# cat_dog_img = Image.open(cat_dog_img)
# cat_dog_img = transform(cat_dog_img)

# cat_dog_img = cat_dog_img.unsqueeze(0)
# heatmap,pred = compute_heatmap(vgg_model,cat_dog_img)
# upsampled_map = upsampleHeatmap(heatmap, cat_dog_img)
# print(f"Prediction: {pred}")

# display_images(upsampled_map, cat_dog_img)



# ---------------------------------------------------------------------------------------------------------------------------------

# Import necessary packages and libraries
import torchvision
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt
import os
import glob

# Load pre-trained model
vgg_model = torchvision.models.vgg16(pretrained=True)
# vgg_model = torchvision.models.efficientnet_b7(pretrained=True)

# Transformation for passing image into the network
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Selecting layers from the model to generate activations
image_to_heatmaps = nn.Sequential(*list(vgg_model.features[:-4]))

def compute_heatmap(model, img):
    model.eval()
    # Compute logits from the model
    logits = model(img)
    # Model's prediction
    pred = logits.max(-1)[-1]
    # Activations from the model
    activations = image_to_heatmaps(img)
    # Compute gradients with respect to the model's most confident prediction
    logits[0, pred].backward(retain_graph=True)
    # Average gradients of the feature map
    pool_grads = model.features[-3].weight.grad.data.mean((0, 2, 3))
    # Multiply each activation map with corresponding gradient average
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pool_grads[i]
    # Calculate mean of weighted activations
    heatmap = torch.mean(activations, dim=1)[0].cpu().detach()
    return heatmap, pred

def upsampleHeatmap(map, image):
    # Permute image
    image = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    # Maximum and minimum value from heatmap
    m, M = map.min(), map.max()
    # Normalize the heatmap
    map = 255 * ((map - m) / (M - m))
    map = np.uint8(map)
    # Resize the heatmap to the same as the input
    map = cv2.resize(map, (224, 224))
    map = cv2.applyColorMap(255 - map, cv2.COLORMAP_JET)
    # Blend heatmap and image
    superimposed = np.uint8(map * 0.7 + image * 0.3)
    return superimposed

def save_images(original_image, heatmap, superimposed_image, save_dir, image_name):
    # Ensure the output folder exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Save images
    original_image.save(os.path.join(save_dir, f"{image_name}_original.png"))
    heatmap_img = Image.fromarray(heatmap)
    heatmap_img.save(os.path.join(save_dir, f"{image_name}_heatmap.png"))
    superimposed_img = Image.fromarray(superimposed_image)
    superimposed_img.save(os.path.join(save_dir, f"{image_name}_superimposed.png"))

def process_multiple_images(input_dir, output_dir):
    # Iterate over each image in the directory
    for i, img_path in enumerate(glob.glob(os.path.join(input_dir, "*.png"))):  # Adjust extension if necessary
        
        if i < 30:
            # Load and transform image
            img_name = os.path.basename(img_path).split('.')[0]  # Get image name without extension
            img = Image.open(img_path)
            transformed_img = transform(img).unsqueeze(0)

            # Compute heatmap
            heatmap, pred = compute_heatmap(vgg_model, transformed_img)

            # Upsample the heatmap
            upsampled_map = upsampleHeatmap(heatmap, transformed_img)
            print(f"Processed {img_name} | Prediction: {pred}")

            # Save images in a corresponding directory within the output folder
            save_path = os.path.join(output_dir, img_name)
            save_images(img, upsampled_map, upsampled_map, save_path, img_name)

# Example usage
input_folder = "./ISTD_Dataset/test/test_A"
output_folder = "./output_images"
process_multiple_images(input_folder, output_folder)





