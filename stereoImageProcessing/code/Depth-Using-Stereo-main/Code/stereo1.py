import matplotlib

from Utils.FundamentalEssentialMatrix import *
from Utils.GeometryUtils import *
from Utils.ImageUtils import *
from Utils.MathUtils import *
from Utils.MiscUtils import *
from Utils.Triangulation import *

matplotlib.use('Agg')
import argparse
from matplotlib import pyplot as plt


def compute_disparity_gpu(left_img, right_img, block_size=11):
    left_cuda = cv2.cuda_GpuMat()
    right_cuda = cv2.cuda_GpuMat()
    left_cuda.upload(left_img)
    right_cuda.upload(right_img)

    stereo = cv2.cuda.createStereoBM(numDisparities=64, blockSize=block_size)
    disparity_cuda = stereo.compute(left_cuda, right_cuda)
    disparity = disparity_cuda.download()
    return disparity


def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--DataPath', default='../Data/Project 3/Dataset 2',
                        help='data path of project3, Default:../Data/Project 3/Dataset 2')
    Parser.add_argument('--DataNumber', type=int, default=2, help='data set number to use the corresponding parameters')
    Args = Parser.parse_args()
    folder_name = Args.DataPath
    dataset_number = Args.DataNumber

    # Camera Intrinsics
    camera_params = {
        1: (np.array([[5299.313, 0, 1263.818], [0, 5299.313, 977.763], [0, 0, 1]]), 177.288),
        2: (np.array([[4396.869, 0, 1353.072], [0, 4396.869, 989.702], [0, 0, 1]]), 144.049),
        3: (np.array([[5806.559, 0, 1429.219], [0, 5806.559, 993.403], [0, 0, 1]]), 174.019)
    }

    if dataset_number not in camera_params:
        print("Invalid dataset number")
        return

    K1, baseline = camera_params[dataset_number]
    f = K1[0, 0]

    # Load images
    images = readImageSet(folder_name, 2)
    image0, image1 = images[0], images[1]

    # Convert to grayscale
    gray0, gray1 = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY), cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

    # Feature extraction using GPU-accelerated SIFT
    sift = cv2.cuda.SIFT_create()
    kp1, des1 = sift.detectAndCompute(cv2.cuda_GpuMat().upload(gray0), None)
    kp2, des2 = sift.detectAndCompute(cv2.cuda_GpuMat().upload(gray1), None)

    # Feature matching using FLANN
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.cuda.DescriptorMatcher_createBFMatcher(cv2.NORM_L2)
    matches = flann.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)[:100]
    matched_pairs = siftFeatures2Array(matches, kp1, kp2)

    # Estimate Essential Matrix
    F_best, matched_pairs_inliers = getInliers(matched_pairs)
    E = getEssentialMatrix(K1, K1, F_best)
    R2, C2 = ExtractCameraPose(E)
    pts3D_4 = get3DPoints(K1, K1, matched_pairs_inliers, R2, C2)

    # Rectification
    h, w = gray0.shape
    _, H1, H2 = cv2.stereoRectifyUncalibrated(np.float32(matched_pairs[:, :2]),
                                              np.float32(matched_pairs[:, 2:]),
                                              F_best, (w, h))
    img1_rectified = cv2.warpPerspective(gray0, H1, (w, h))
    img2_rectified = cv2.warpPerspective(gray1, H2, (w, h))

    # Compute disparity using GPU
    disparity_map = compute_disparity_gpu(img1_rectified, img2_rectified)
    disparity_map = np.uint8(disparity_map * 255 / np.max(disparity_map))

    # Compute depth
    depth = (baseline * f) / (disparity_map + 1e-10)
    depth[depth > 100000] = 100000
    depth_map = np.uint8(depth * 255 / np.max(depth))

    # Save results
    plt.imsave(f'../Results/disparity_image_{dataset_number}.png', disparity_map, cmap='hot')
    plt.imsave(f'../Results/depth_image_{dataset_number}.png', depth_map, cmap='hot')


if __name__ == "__main__":
    main()
