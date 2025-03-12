import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.optimize
import torch
import torchvision
import torchvision.transforms.functional as tvtf
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights

FocalLength = 0.673624570232178
weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=weights)
model.load_state_dict(torch.load("MaskrCNN_model.pt"))
model.eval()

centre = None
COLOURS = [
    tuple(int(colour_hex.strip('#')[i:i + 2], 16) for i in (0, 2, 4))
    for colour_hex in plt.rcParams['axes.prop_cycle'].by_key()['color']
]


def imageColorChange(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def preprocess_image(image):
    image = tvtf.to_tensor(image)
    image = image.unsqueeze(dim=0)
    return image


def get_detections(maskrcnn, imgs, score_threshold=0.5):  # person, dog, elephant, zebra, giraffe
    det = []
    lbls = []
    scores = []
    masks = []

    for img in imgs:
        with torch.no_grad():
            result = maskrcnn(preprocess_image(img))[0]
        mask = result["scores"] > score_threshold
        det.append(result["boxes"][mask].detach().cpu().numpy())
        lbls.append(result["labels"][mask].detach().cpu().numpy())
        scores.append(result["scores"][mask].detach().cpu().numpy())
        masks.append(result["masks"][mask])
    return det, lbls, scores, masks


def draw_detections(img, det, colours=None, obj_order=None):
    # i starts from 0, len(det), (tlx, tly, brx, bry) are position
    if colours is None:
        colours = COLOURS
    for i, (tlx, tly, brx, bry) in enumerate(det):
        if obj_order is not None and len(obj_order) < i:
            i = obj_order[i]
        i %= len(colours)
        cv2.rectangle(img, (tlx, tly), (brx, bry), color=colours[i], thickness=2)


def tlbr_to_center1(boxes):
    points = []
    for tlx, tly, brx, bry in boxes:
        cx = (tlx + brx) / 2
        cy = (tly + bry) / 2
        points.append([cx, cy])
    return points


def tlbr_to_corner(boxes):
    points = []
    for tlx, tly, brx, bry in boxes:
        cx = (tlx + tlx) / 2
        cy = (tly + tly) / 2
        points.append((cx, cy))
    return points


def tlbr_to_corner_br(boxes):
    points = []
    for tlx, tly, brx, bry in boxes:
        cx = (brx + brx) / 2
        cy = (bry + bry) / 2
        points.append((cx, cy))
    return points


def tlbr_to_area(boxes):
    areas = []
    for tlx, tly, brx, bry in boxes:
        cx = (brx - tlx)
        cy = (bry - tly)
        areas.append(abs(cx * cy))
    return areas


def get_horiz_dist_centre(boxes):
    pnts1 = np.array(tlbr_to_center1(boxes[0]))[:, 0]
    pnts2 = np.array(tlbr_to_center1(boxes[1]))[:, 0]
    return pnts1[:, None] - pnts2[None]


def get_horiz_dist_corner_tl(boxes):
    pnts1 = np.array(tlbr_to_corner(boxes[0]))[:, 0]
    pnts2 = np.array(tlbr_to_corner(boxes[1]))[:, 0]
    return pnts1[:, None] - pnts2[None]


def get_horiz_dist_corner_br(boxes):
    pnts1 = np.array(tlbr_to_corner_br(boxes[0]))[:, 0]
    pnts2 = np.array(tlbr_to_corner_br(boxes[1]))[:, 0]
    return pnts1[:, None] - pnts2[None]


def get_vertic_dist_centre(boxes):
    pnts1 = np.array(tlbr_to_center1(boxes[0]))[:, 1]
    pnts2 = np.array(tlbr_to_center1(boxes[1]))[:, 1]
    return pnts1[:, None] - pnts2[None]


def get_area_diffs(boxes):
    pnts1 = np.array(tlbr_to_area(boxes[0]))
    pnts2 = np.array(tlbr_to_area(boxes[1]))
    return abs(pnts1[:, None] - pnts2[None])


def get_dist_to_centre_tl(box, cntr=centre):
    pnts = np.array(tlbr_to_corner(box))[:, 0]
    return abs(pnts - cntr)


def get_dist_to_centre_br(box, cntr=centre):
    pnts = np.array(tlbr_to_corner_br(box))[:, 0]
    return abs(pnts - cntr)


def get_cost(boxes, lbls=None, sz1=400):
    alpha = sz1
    beta = 10
    gamma = 5

    # vertical_dist, scale by gamma since can't move up or down
    vert_dist = gamma * abs(get_vertic_dist_centre(boxes))

    # horizontal distance.
    horiz_dist = get_horiz_dist_centre(boxes)

    # increase cost if object has moved from right to left.
    horiz_dist[horiz_dist < 0] = beta * abs(horiz_dist[horiz_dist < 0])

    # area of box
    area_diffs = get_area_diffs(boxes) / alpha

    cost = np.array([vert_dist, horiz_dist, area_diffs])

    cost = cost.sum(axis=0)

    # add penalty term for different object classes
    if lbls is not None:
        for i in range(cost.shape[0]):
            for j in range(cost.shape[1]):
                if lbls[0][i] != lbls[1][j]:
                    cost[i, j] += 150
    return cost


def annotate_class(img, det, class_map, conf=None, colours=None):
    if colours is None:
        colours = COLOURS
    for i, (tlx, tly, brx, bry) in enumerate(det):
        txt = class_map[i]
        if conf is not None:
            txt += f' {conf[i]:1.3f}'
        offset = 1
        cv2.rectangle(img,
                      (tlx - offset, tly - offset + 12),
                      (tlx - offset + len(txt) * 12, tly),
                      color=colours[i % len(colours)],
                      thickness=cv2.FILLED)

        ff = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(img, txt, (tlx, tly - 1 + 12), fontFace=ff, fontScale=1.0, color=(255,) * 3)


def processing_images(left_img, right_img, startTime):
    global centre
    left_img = imageColorChange(left_img)
    right_img = imageColorChange(right_img)

    # Stereo image dimensions
    sz1, sz2 = right_img.shape[1], right_img.shape[0]
    centre = sz1 / 2
    # Preprocess images for model input

    imgs = [left_img, right_img]

    # Get detections from the model
    det, lbls, _, _ = get_detections(model, imgs)

    # Calculate costs between detected objects
    cost = get_cost(det, lbls)

    # Perform linear sum assignment to get tracks
    tracks = scipy.optimize.linear_sum_assignment(cost)

    dists_tl = get_horiz_dist_corner_tl(det)
    dists_br = get_horiz_dist_corner_br(det)

    # Determine final distances based on closest corner to center
    final_dists = []
    dctl = get_dist_to_centre_tl(det[0], centre)
    dcbr = get_dist_to_centre_br(det[0], centre)

    for i, j in zip(*tracks):
        if dctl[i] < dcbr[i]:
            final_dists.append((dists_tl[i][j], np.array(weights.meta["categories"])[lbls[0]][i]))

        else:
            final_dists.append((dists_br[i][j], np.array(weights.meta["categories"])[lbls[0]][i]))

    tanTheta = (1 / (28.2 - FocalLength)) * (7.05 / 2) * sz1 / 227.710

    fd = [i for (i, j) in final_dists]

    # find the distance away
    dists_away = (7.05 / 2) * sz1 * (1 / tanTheta) / np.array(fd) + FocalLength

    cat_dist = []
    for i in range(len(dists_away)):
        cat_dist.append(f'{np.array(weights.meta["categories"])[lbls[0]][i]} {dists_away[i]:.1f}cm')

    t1 = [list(tracks[1]), list(tracks[0])]
    for i, imgi in enumerate(imgs[:1]):
        deti = det[i].astype(np.int32)
        draw_detections(imgi, deti[list(tracks[i])], obj_order=list(t1[i]))
        annotate_class(imgi, deti[list(tracks[i])], cat_dist)
    print('Time take:', (time.time() - startTime))
    return imgs


cv2.namedWindow("Image Depth from Left camera", cv2.WINDOW_NORMAL)
for i in range(0, 5):
    if i in [0, 4]:
        images = processing_images(
            cv2.imread(f'StereoCameraImages/LeftImages/left_image_0{i}.jpg'),
            cv2.imread(f'StereoCameraImages/RightImages/right_image_0{i}.jpg'), time.time())
        cv2.imshow("Image Depth from Left camera", images[0])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
