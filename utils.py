import math
import cv2
import numpy as np
import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F

def classify_hand(mp_hands, hand_landmarks, image_width):
    """
    Classify the hand as left or right based on the x-coordinates of the thumb and index finger base.

    Parameters:
    - hand_landmarks: The hand landmarks object from MediaPipe.
    - image_width, image_height: Dimensions of the image to convert normalized coordinates into pixel coordinates.

    Returns:
    - A string indicating whether the hand is 'Left' or 'Right'.
    """
    # Convert normalized coordinates to pixel coordinates for thumb tip and index base
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_base = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]

    thumb_tip_x_px = thumb_tip.x * image_width
    index_base_x_px = index_base.x * image_width

    # Determine orientation
    if thumb_tip_x_px > index_base_x_px:
        return 'Right'  # For the viewer, this appears as a left hand, but it's actually the right hand from the person's perspective
    else:
        return 'Left'  # Similarly, this appears as a right hand but it's the left hand from the person's perspective


def align_hand(hands, mp_hands, img_path):

    # Load an image.
    image = cv2.imread(img_path)

    # Convert the image color from BGR to RGB.
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and detect hands.
    results = hands.process(image_rgb)


    if results.multi_hand_landmarks:
        # Assuming you have the image's dimensions
        image_width, image_height = image.shape[1], image.shape[0]

        for hand_landmarks in results.multi_hand_landmarks:
            # Extract MCP joints for Index, Middle, Ring, and Pinky fingers, as well as the Wrist and Thumb MCP
            index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
            middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
            ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
            pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC]  # Note: Using THUMB_CMC for Thumb MCP

            # Convert normalized coordinates to pixel coordinates
            index_mcp_px = (int(index_mcp.x * image_width), int(index_mcp.y * image_height))
            middle_mcp_px = (int(middle_mcp.x * image_width), int(middle_mcp.y * image_height))
            ring_mcp_px = (int(ring_mcp.x * image_width), int(ring_mcp.y * image_height))
            pinky_mcp_px = (int(pinky_mcp.x * image_width), int(pinky_mcp.y * image_height))
            wrist_px = (int(wrist.x * image_width), int(wrist.y * image_height))
            thumb_mcp_px = (int(thumb_mcp.x * image_width), int(thumb_mcp.y * image_height))

            hand_orientation = classify_hand(mp_hands, hand_landmarks, image_width)
        points = np.array([
            index_mcp_px,
            middle_mcp_px,
            ring_mcp_px,
            pinky_mcp_px,
            (wrist_px[0],pinky_mcp_px[1]),
            wrist_px,
            thumb_mcp_px
        ], dtype=np.int32)
    else:
       return "No Hand Detected", "No Matrix", "No Points"
    # Example coordinates for Index Finger MCP and Pinky MCP in pixels
    x_index, y_index = index_mcp.x, index_mcp.y  # Replace with actual pixel coordinates of Index Finger MCP
    x_pinky, y_pinky = pinky_mcp.x, pinky_mcp.y  # Replace with actual pixel coordinates of Pinky MCP

    distance = math.sqrt( (y_index - y_pinky)**2 + (x_index - x_pinky)**2 )
    unit_vector = (((x_pinky - x_index)/distance), ((y_pinky - y_index)/distance))


    angle_with_horizontal = math.atan2(unit_vector[1], unit_vector[0])
    angle_with_horizontal = angle_with_horizontal * (180 / math.pi)

    rotation_angle = 0

    if hand_orientation == "Right":
      if -180 <= angle_with_horizontal <= -90:
        rotation_angle = angle_with_horizontal + 180
      elif 0 <= angle_with_horizontal <= 180:
        rotation_angle = angle_with_horizontal - 180
      else:
        rotation_angle = 180 - angle_with_horizontal

    else:
      if -180 <= angle_with_horizontal <= -90:
        rotation_angle = angle_with_horizontal + 180
      elif -90 <= angle_with_horizontal <= 180:
        rotation_angle = angle_with_horizontal

    image_height, image_width = image.shape[:2]

    # Center of rotation - we'll rotate the image around its center
    center = (image_width // 2, image_height // 2)

    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)

    # Perform the rotation
    rotated_image = cv2.warpAffine(image, rotation_matrix, (image_width, image_height))

    return rotated_image, rotation_matrix, points

def extract_roi(hands, mp_hands, img_path):

    rotated_image, rotation_matrix ,points = align_hand(hands, mp_hands,img_path)
    if type(points) == str:
        return "No Hand Detected"
    ones = np.ones(shape=(len(points), 1))
    points_ones = np.hstack([points, ones])

    # Apply the rotation matrix to the points
    new_points = cv2.transform(np.array([points_ones]), rotation_matrix)

    # The result is a 3D array, convert it back to 2D
    new_points = new_points.squeeze()

    new_points = np.round(new_points).astype(np.int32)

    x, y, w, h = cv2.boundingRect(new_points)

    # Crop the image using the bounding rectangle
    cropped_image = rotated_image[y:y+h, x:x+w]

    return cropped_image


def loadCNNModel(weightPath, device):
    """
    @weightPath: path to the ROILAnet() weights
    loads localization network with pretrained weights
    """
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1321)
    model.load_state_dict(torch.load(weightPath, map_location=torch.device(device)))
    model.fc = nn.Identity()
    model.to(device)
    return model


def calculate_cosine_similarity(output1, output2):

    cosine_sim = F.cosine_similarity(output1, output2, dim=0)

    return cosine_sim.item()
