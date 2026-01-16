
import os
import numpy as np
from PIL import Image
import cv2
from time import time
from torchvision import transforms
import onnxruntime as ort
import argparse

# ------------------------------------------------------------------------------
# ARGUMENTS
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser('parser for face detector inference')
parser.add_argument('--image_folder', type=str, default='test_data', help='provide image path')
parser.add_argument('--detector', type=str, default='onnx_models/RFB_finetuned_with_postprocessing.onnx', help='face detector path')
parser.add_argument('--landmark', type=str, default='onnx_models/landmark_model.onnx', help='landmark model path')
parser.add_argument('--output_folder', type=str, default='crop_output_buffalo_style', help='output folder')
parser.add_argument("--threshold", type=float, default=0.8, help='probability threshold')
parser.add_argument("--iou_threshold", type=float, default=0.5, help='IOU threshold for NMS')
args = parser.parse_args()

# ------------------------------------------------------------------------------
# MODEL LOADING
# ------------------------------------------------------------------------------
so = ort.SessionOptions()
session = ort.InferenceSession(args.detector, sess_options=so, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
session_lm = ort.InferenceSession(args.landmark, sess_options=so, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

transform = transforms.Compose([
    transforms.Resize((360, 480)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# ------------------------------------------------------------------------------
# TEMPLATE
# ------------------------------------------------------------------------------
TEMPLATE = np.float32([
    (0.0792396913815, 0.339223741112), (0.0829219487236, 0.456955367943),
    (0.0967927109165, 0.575648016728), (0.122141515615, 0.691921601066),
    (0.168687863544, 0.800341263616), (0.239789390707, 0.895732504778),
    (0.325662452515, 0.977068762493), (0.422318282013, 1.04329000149),
    (0.531777802068, 1.06080371126), (0.641296298053, 1.03981924107),
    (0.738105872266, 0.972268833998), (0.824444363295, 0.889624082279),
    (0.894792677532, 0.792494155836), (0.939395486253, 0.681546643421),
    (0.96111933829, 0.562238253072), (0.970579841181, 0.441758925744),
    (0.971193274221, 0.322118743967), (0.163846223133, 0.249151738053),
    (0.21780354657, 0.204255863861), (0.291299351124, 0.192367318323),
    (0.367460241458, 0.203582210627), (0.4392945113, 0.233135599851),
    (0.586445962425, 0.228141644834), (0.660152671635, 0.195923841854),
    (0.737466449096, 0.182360984545), (0.813236546239, 0.192828009114),
    (0.8707571886, 0.235293377042), (0.51534533827, 0.31863546193),
    (0.516221448289, 0.396200446263), (0.517118861835, 0.473797687758),
    (0.51816430343, 0.553157797772), (0.433701156035, 0.604054457668),
    (0.475501237769, 0.62076344024), (0.520712933176, 0.634268222208),
    (0.565874114041, 0.618796581487), (0.607054002672, 0.60157671656),
    (0.252418718401, 0.331052263829), (0.298663015648, 0.302646354002),
    (0.355749724218, 0.303020650651), (0.403718978315, 0.33867711083),
    (0.352507175597, 0.349987615384), (0.296791759886, 0.350478978225),
    (0.631326076346, 0.334136672344), (0.679073381078, 0.29645404267),
    (0.73597236153, 0.294721285802), (0.782865376271, 0.321305281656),
    (0.740312274764, 0.341849376713), (0.68499850091, 0.343734332172),
    (0.353167761422, 0.746189164237), (0.414587777921, 0.719053835073),
    (0.477677654595, 0.706835892494), (0.522732900812, 0.717092275768),
    (0.569832064287, 0.705414478982), (0.635195811927, 0.71565572516),
    (0.69951672331, 0.739419187253), (0.639447159575, 0.805236879972),
    (0.576410514055, 0.835436670169), (0.525398405766, 0.841706377792),
    (0.47641545769, 0.837505914975), (0.41379548902, 0.810045601727),
    (0.380084785646, 0.749979603086), (0.477955996282, 0.74513234612),
    (0.523389793327, 0.748924302636), (0.571057789237, 0.74332894691),
    (0.672409137852, 0.744177032192), (0.572539621444, 0.776609286626),
    (0.5240106503, 0.783370783245), (0.477561227414, 0.778476346951)
])

TPL_MIN, TPL_MAX = np.min(TEMPLATE, axis=0), np.max(TEMPLATE, axis=0)
MINMAX_TEMPLATE = (TEMPLATE - TPL_MIN) / (TPL_MAX - TPL_MIN)
INNER_EYES_AND_BOTTOM_LIP = [39, 42, 57]

# ------------------------------------------------------------------------------
# CORE LOGIC
# ------------------------------------------------------------------------------

def detect(image_array):
    input_image_name = session.get_inputs()[0].name
    inputs = {
        input_image_name: image_array,
        'iou_threshold': np.array([args.iou_threshold], dtype=np.float32),
        'score_threshold': np.array([args.threshold], dtype=np.float32)
    }
    start = time()
    boxes, labels, probs = session.run(None, inputs)
    end = time()
    return boxes, probs, end - start

def production_cropping_buffalo(img, bbox, margin=20, scale=0.65, shift_y=0.05):
    """
    Buffalo Style Logic:
    1. USES Landmarks for Rotation/Scaling (Alignment).
    2. BUT prevents skewing by using Similarity Transform (estimateAffinePartial2D) 
       instead of standard Affine.
    3. Scale 0.65 provides the "loose" crop similar to Buffalo.
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox[:4].astype(int)
    
    # 1. Expand Bbox (Margin)
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(w, x2 + margin)
    y2 = min(h, y2 + margin)

    ch, cw = y2 - y1, x2 - x1
    
    if x1 >= x2 or y1 >= y2: return None
    face_crop = img[y1:y2, x1:x2]
    if face_crop.size == 0: return None

    # 2. Landmark Inference
    scaler = np.array([ch, cw])
    crop_resized = cv2.resize(face_crop, (64, 64))
    crop_resized = np.expand_dims(crop_resized, axis=0).astype(np.uint8)
    
    inputs = {session_lm.get_inputs()[0].name: crop_resized}
    ort_outputs = session_lm.run(None, inputs)
    keypoints = np.array(ort_outputs).reshape(98, 2)
    
    landmarks = (keypoints * scaler) + (y1, x1)
    
    landmarks_xy = []
    lm_cnt = 0
    for y, x in landmarks:
        lm_cnt += 1
        if(lm_cnt == 65 or lm_cnt == 69 or lm_cnt == 86):
            landmarks_xy.append([x, y])

    npLandmarks = np.float32(landmarks_xy)    
    landmarkIndices = INNER_EYES_AND_BOTTOM_LIP
    npLandmarkIndices = np.array(landmarkIndices)

    # 3. Create Centered Template
    T = MINMAX_TEMPLATE[npLandmarkIndices].copy()
    
    # "Buffalo Style" Centering & Scaling
    # scale=0.65 is looser (face occupies ~65% of image)
    T = (T - 0.5) * scale + 0.5
    T[:, 1] += shift_y
    T[:, 0] *= 112
    T[:, 1] *= 112
    
    # -------------------------------------------------------------------------
    # CRITICAL FIX FOR SKEW: estimateAffinePartial2D
    # -------------------------------------------------------------------------
    # Standard getAffineTransform allows shearing (skewing) to force fit.
    # estimateAffinePartial2D restricts the transform to Rotation + Scale + Translation.
    # It finds the "best fit" without destroying the face geometry.
    # print(f"Margin and scale is: {margin} : {scale}")
    try:
        H, _ = cv2.estimateAffinePartial2D(npLandmarks, T, method=cv2.LMEDS) # COMMENTING THIS
        # H = cv2.getAffineTransform(npLandmarks, T)
        if H is None:
            # Fallback if points are collinear or estimation fails
            print("Robust alignment failed, falling back to simple affine.")
            H = cv2.getAffineTransform(npLandmarks, T)
            
        return cv2.warpAffine(img, H, (112, 112))
        
    except Exception as e:
        print(f"Alignment error: {e}")
        return cv2.resize(face_crop, (112, 112))

def detect_landmark_and_align(ori_img_path, output_folder):
    filename_base = os.path.splitext(os.path.basename(ori_img_path))[0]
    img_bgr = cv2.imread(ori_img_path)
    if img_bgr is None: return

    image_h, image_w, _ = img_bgr.shape
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    
    transformed_image = transform(img_pil)
    image_array = np.expand_dims(np.array(transformed_image), axis=0).astype(np.float32)

    boxes, score, _ = detect(image_array)

    if boxes is None: return

    for i, bbox in enumerate(boxes):
        x_min, y_min, x_max, y_max = map(float, bbox)
        x_min = x_min * image_w if x_min <= 1 else x_min
        y_min = y_min * image_h if y_min <= 1 else y_min
        x_max = x_max * image_w if x_max <= 1 else x_max
        y_max = y_max * image_h if y_max <= 1 else y_max

        bbox_np = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
        
        # Scale 0.65 = Buffalo Style (Loose)
        final_img = production_cropping_buffalo(img_bgr, bbox_np, margin=25, scale=0.65, shift_y=0.18) # DO NOT CHANGE THESE VALUES

        if final_img is not None:
            save_path = os.path.join(output_folder, f"{filename_base}_face_{i}.jpg")
            cv2.imwrite(save_path, final_img)
            print(f"Saved aligned face: {save_path}")

def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(root, file)
                rel_path = os.path.relpath(root, input_folder)
                target_dir = os.path.join(output_folder, rel_path)
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                detect_landmark_and_align(image_path, target_dir)

if __name__ == "__main__":
    process_folder(args.image_folder, args.output_folder)