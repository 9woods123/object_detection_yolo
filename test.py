from ultralytics import YOLO
import os

# Load a pretrained YOLO model
model = YOLO("best.pt")  # Replace with your model file

# Directory containing the images
image_dir = "train_data/images/val"

# Output directory for results
output_dir = "predict_result"
os.makedirs(output_dir, exist_ok=True)

# List all image files in the directory
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Process each image
for number, image_file in enumerate(image_files, start=1):
    image_path = os.path.join(image_dir, image_file)
    results = model(image_path)  # Run inference on the image

    for result in results:
        result.show()  # Display the result on screen (optional)
        
        # Save the result to disk
        output_path = os.path.join(output_dir, f"result_{number}.jpg")
        result.save(filename=output_path)

print(f"Processed {len(image_files)} images. Results saved in '{output_dir}'.")
