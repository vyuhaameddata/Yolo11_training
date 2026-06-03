import cv2
import numpy as np
from ultralytics import YOLO
import os

# # Load model
# model = YOLO(r'w\segment.pt')

# # Run prediction
# results = model.predict(
#     r"D:\Shreyan\Development\Train\Data\cell\yolo11_data\crop\test\org",
#     save=True, conf=0.2, project=r'D:\Shreyan\Development\Train\Data\cell\yolo11_data\crop\test', name='mask',
#     show_labels=False, show_boxes=False, show_conf=False
# )

# # Create directories for outputs
# save_dir_poly = r"D:\Shreyan\Development\Train\Data\cell\yolo11_data\crop\test\poly"
# save_dir_mask = r"D:\Shreyan\Development\Train\Data\cell\yolo11_data\crop\test\binary_mask"
# os.makedirs(save_dir_poly, exist_ok=True)
# os.makedirs(save_dir_mask, exist_ok=True)

# # Define kernel for morphological operations
# kernel = np.ones((9, 9), np.uint8)  # You can adjust the size for stronger or weaker effects

# for i, result in enumerate(results):
#     # Initialize a combined mask (single channel)
#     combined_mask = np.zeros_like(result.orig_img[:, :, 0])

#     # Extract masks and draw polygons
#     if result.masks is not None:
#         for j, mask in enumerate(result.masks.xy):
#             print(f"Polygon {j} for Image {i}: {mask}")
#             points = np.array(mask, dtype=np.int32)

#             # Create and save binary mask for each polygon
#             mask_img = np.zeros_like(combined_mask)
#             cv2.fillPoly(mask_img, [points], color=255)

#             # Apply morphological operations to reduce noise
#             mask_img = cv2.erode(mask_img, kernel, iterations=1)
#             mask_img = cv2.dilate(mask_img, kernel, iterations=1)

#             mask_path = os.path.join(save_dir_mask, f"mask_{i}_poly_{j}.png")
#             cv2.imwrite(mask_path, mask_img)
#             print(f"Saved Binary Mask: {mask_path}")

#             # Add polygon to combined mask
#             cv2.fillPoly(combined_mask, [points], color=255)

#         # Apply erosion and dilation to the combined mask
#         combined_mask = cv2.erode(combined_mask, kernel, iterations=1)
#         combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)

#         # Convert to 3-channel for visualization
#         combined_mask_vis = cv2.merge([combined_mask, combined_mask, combined_mask])

#         # Draw all polygons on the combined mask for visualization
#         k = 1
#         for mask in result.masks.xy:
#             save_path = os.path.join(save_dir_poly, f"combined_mask_poly_{i}_{k}.png")
#             k += 1
#             cv2.imwrite(save_path, combined_mask_vis)
#             points = np.array(mask, dtype=np.int32)
#             cv2.polylines(combined_mask_vis, [points], isClosed=True, color=(0, 255, 0), thickness=1)

#         # Save combined mask with polygons visualization
#         save_path = os.path.join(save_dir_poly, f"combined_mask_poly_{i}.png")
#         cv2.imwrite(save_path, combined_mask_vis)
#         print(f"Saved Combined Mask with Polygons: {save_path}")





# Process results list
# for result in results:
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs
#     obb = result.obb  # Oriented boxes object for OBB outputs
#     result.show()  # display to screen
#     result.save(filename="result.jpg")  # save to disk


...








# Load model
model = YOLO(r'w\segment.pt')

# Run prediction
results = model.predict(
    r"D:\Shreyan\Development\Train\Data\cell\yolo11_data\crop\tif",
    conf=0.2
)

# Visualize polygons without masks
save_dir = r"D:\Shreyan\Development\Train\Data\cell\yolo11_data\crop\bwpoly"
os.makedirs(save_dir, exist_ok=True)

for i, result in enumerate(results):
    img = result.orig_img.copy()
    img_name = f"result_{i}.png"  # You can customize this further

    # Extract masks and draw polygons
    if result.masks is not None:
        for mask in result.masks.xy:
            points = np.array(mask, dtype=np.int32)
            cv2.polylines(img, [points], isClosed=True, color=(255, 0, 255), thickness=1)

    # Save the result
    save_path = os.path.join(save_dir, img_name)
    cv2.imwrite(save_path, img)
    print(f"Saved: {save_path}")


# # Load model
# model = YOLO(r'w\segment.pt')

# # Run prediction
# results = model.predict(
#     r"D:\Shreyan\Development\Train\Data\cell\yolo11_data\crop\tif",
#     save=True, conf=0.2, project=r'D:\Shreyan\Development\Train\Data\cell\yolo11_data\crop', name='predict',
# )