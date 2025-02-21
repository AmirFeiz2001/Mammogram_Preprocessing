import os
import matplotlib.pyplot as plt
from datetime import datetime

def visualize_results(org_image, painted_org_image, enhanced_image, painted_image, output_dir="output/plots"):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig = plt.figure(figsize=(10, 20))
    fig.add_subplot(1, 2, 1)
    plt.imshow(painted_org_image, cmap='gray')
    plt.title("Original Image with Chaincode")
    plt.axis('off')

    fig.add_subplot(1, 2, 2)
    plt.imshow(painted_image, cmap='gray')
    plt.title("Preprocessed Image with Bounding Box")
    plt.axis('off')

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"result_{timestamp}.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Visualization saved to {output_path}")

def visualize_steps(preprocessor, output_dir="output/plots"):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig = plt.figure(figsize=(30, 40))
    steps = [
        ("Original Image", preprocessor._org_image),
        ("Cropped Image", preprocessor._crop_image()),
        ("Text Removed Image", preprocessor._remove_text()),
        ("Enhanced Image", preprocessor._enhance_image())
    ]

    for i, (title, image) in enumerate(steps, 1):
        fig.add_subplot(1, 5, i)
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.axis('off')

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"preprocessing_steps_{timestamp}.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Preprocessing steps saved to {output_path}")
