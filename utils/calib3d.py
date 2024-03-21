import cv2
import numpy as np
import os
from PIL import Image, ImageOps
from scipy.optimize import least_squares, minimize
from sklearn.linear_model import RANSACRegressor, LinearRegression

euclidean_err = lambda y_true, y_pred: np.linalg.norm(y_true - y_pred, axis=1)


def Rodrigues_R2V(R, v):
    tmp = (R - R.T) / 2.
    print()


def save_to_jpg(image, save_path):
    # imagedata = image.read()
    # image.close()
    # pngImage = Image.frombytes(imagedata)
    pngImage = Image.open(image)
    # pngImage = ImageOps.exif_transpose(pngImage)
    pngImage.convert("RGB").save(save_path, "JPEG", quality=100)
    return save_path


def read_camera_matrix_3dscanner(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    K = []
    for i in range(3):
        K.append([float(v) for v in lines[i].split()[:3]])
    K = np.array(K, dtype=np.float32)
    return K


def read_pose_3dscanner(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    pose = []
    for i in range(3):
        pose.append([float(value) for value in lines[i].split()[:4]])
    pose = np.array(pose, dtype=np.float32)
    return pose

def image_transform(image: np):
    (height, width) = image.shape[:2]
    if height >= width:
        return image
    # rotate the src image 90 degrees clockwise
    rotated_image = cv2.transpose(image)
    # rotated_image = cv2.flip(rotated_image, 1)
    return rotated_image


def pixel2cam(points, K):
    camP = []
    for p in points:
        camP.append(((points[0] - K[0, 2]) / K[0, 0], (points[1] - K[1, 2]) / K[1, 1]))
    return np.array(camP, dtype=np.float32)


def proj_3dto2d(params, points3d):
    """
        params: fx, fy, cx, cy, rx, ry, rz, tx, ty, tz
        points2d: nx2
        points3d: nx3
        """
    fx, fy, cx, cy, rx, ry, rz, tx, ty, tz = params
    # intrinsic
    projection_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    rotation_vector = np.array([[rx], [ry], [rz]])
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    # shift
    shift = np.array([[tx], [ty], [tz]])

    # project to 2d
    projected_points = projection_matrix @ (rotation_matrix @ points3d.T + shift)
    projected_points = cv2.convertPointsFromHomogeneous(projected_points.T).squeeze()

    return projected_points


def reproj_func(params, points2d, points3d):
    """
    params: fx, fy, cx, cy, rx, ry, rz, tx, ty, tz
    points2d: nx2
    points3d: nx3
    """
    projected_points = proj_3dto2d(params, points3d)

    # calculate error
    reprojection_error = euclidean_err(points2d, projected_points)

    return reprojection_error

def triangle_func(params, points1, points2, P1, P2):
    fx1, fy1, cx1, cy1, fx2, fy2, cx2, cy2 = params[:8]
    K1 = np.array([[fx1, 0, cx1], [0, fy1, cy1], [0, 0, 1]])
    K2 = np.array([[fx2, 0, cx2], [0, fy2, cy2], [0, 0, 1]])
    points3d = cv2.triangulatePoints(K1 @ P1, K2 @ P2, points1, points2)
    points3d = cv2.convertPointsFromHomogeneous(points3d.T).squeeze()
    return points3d

def triangle_reproj_func(params, points2d, points1, points2, P1, P2):
    """
    params: K1, K2, K3, rx, ry, rz, tx, ty, tz
    points2d: nx2
    points1/2: nx2
    """
    points3d = triangle_func(params[:8], points1.reshape(-1, 1, 2), points2.reshape(-1, 1, 2), P1, P2)
    return reproj_func(params[8:], points2d, points3d)

def reproject_avg_err(params, points2d, points3d):
    return np.sum(reproj_func(params, points2d, points3d)) / len(points2d)


class ReprojRegressor(LinearRegression):
    def __init__(self, initial_params=np.array([1000, 1000, 500, 500, 0, 0, 0, 0, 0, 0]),
                 bounds=np.array([[0, 3000], [0, 3000], [0, 2000], [0, 2000],
                                  [-10, 10], [-10, 10], [-10, 10],
                                  [-10, 10], [-10, 10], [-10, 10]]), verbose=1, method='lm'):
        self.verbose = verbose
        self.method = method
        self.bounds = bounds
        self.initial_params = initial_params
        self.coef_ = initial_params
        self.success = False

    @staticmethod
    def is_model_valid(model, X, y):
        return model.success

    def fit(self, points3d, points2d, sample_weight=None, nfev=None):
        result = least_squares(reproj_func, self.initial_params, args=(points2d, points3d), verbose=self.verbose,
                               method=self.method, max_nfev=nfev)
        self.coef_ = result.x
        self.success = result.success
        return self

    def score(self, points3d, points2d, sample_weight=None):
        return 1.

    def predict(self, points3d):
        return proj_3dto2d(self.coef_, points3d)

def pose2params(K, R, T):
    R, _ = cv2.Rodrigues(R)
    R = R.squeeze()
    T = T.squeeze()
    return np.array([K[0,0], K[1,1], K[0,2], K[1,2], R[0], R[1], R[2], T[0], T[1], T[2]])

def params2pose(initial_params):
    fx, fy, cx, cy, rx, ry, rz, tx, ty, tz = initial_params
    R = np.array([[rx], [ry], [rz]])
    R, _ = cv2.Rodrigues(R)
    T = np.array([[tx], [ty], [tz]])
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    return K, R, T

def DLS_pose_est(points3d, points2d, initial_params=np.array([1000, 1000, 500, 500, 0, 0, 0, 0, 0, 0]), threshold=None,
                 useRANSAC=True, max_trials=25,
                 bounds=np.array([[0, 3000], [0, 3000], [0, 2000], [0, 2000],
                                  [-10, 10], [-10, 10], [-10, 10],
                                  [-10, 10], [-10, 10], [-10, 10]])):
    '''
    threshold = 8.0 when not useFilter
              = adaptive when useFilter
    '''
    if useRANSAC:
        reg = RANSACRegressor(base_estimator=ReprojRegressor(initial_params, bounds, verbose=1, method='lm'),
                              min_samples=max(np.ceil(len(points2d) * 0.01), 20),
                              loss=euclidean_err, residual_threshold=threshold,
                              is_model_valid=ReprojRegressor.is_model_valid, random_state=0, max_trials=max_trials)
        reg = reg.fit(points3d, points2d)
        # if not reg.estimator_.success:
        #     reg.estimator_ = reg.estimator_.fit(points3d[reg.inlier_mask_], points2d[reg.inlier_mask_], nfev=50000)
        test = reproj_func(reg.estimator_.coef_, points2d, points3d) < 10
        ratio = 1. * np.sum(test) / len(test)
        print(f'valid num:{ratio * len(points2d)}')
        fx, fy, cx, cy, rx, ry, rz, tx, ty, tz = reg.estimator_.coef_
        R = np.array([[rx], [ry], [rz]])
        T = np.array([[tx], [ty], [tz]])
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        return reg.estimator_.success and fx > 0 and fy > 0 and cx > 0 and cy > 0, R, T, K, reg.inlier_mask_
        # return reproject_avg_err(reg.estimator_.coef_, points2d[reg.inlier_mask_], points3d[reg.inlier_mask_]) < 8., R, T, K, reg.inlier_mask_
    else:
        # 进行非线性最小二乘优化
        # result = least_squares(reproj_func, initial_params, bounds=bounds.T, args=(points2d, points3d), verbose=1)
        result = least_squares(reproj_func, initial_params, args=(points2d, points3d), verbose=1, method='lm')
        # result = minimize(reproject_sum_err, initial_params, args=(points2d, points3d), options={'disp':True})
        test = result.fun < 10
        ratio = 1. * np.sum(test) / len(test)
        print(f'valid ratio:{ratio}')
        fx, fy, cx, cy, rx, ry, rz, tx, ty, tz = result.x
        R = np.array([[rx], [ry], [rz]])
        T = np.array([[tx], [ty], [tz]])
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        return result.success and fx > 0 and fy > 0 and cx > 0 and cy > 0, R, T, K, None

def DLS_pose_est_init_params(P1, K3):
    rot_vec1, _ = cv2.Rodrigues(P1[:3, :3])
    rot_vec1 = rot_vec1.flatten()
    shift1 = P1[:3, 3]
    intrinsic3 = np.array([K3[0, 0], K3[1, 1], K3[0, 2], K3[1, 2]])
    init_params = np.hstack((intrinsic3, rot_vec1, shift1))
    return init_params

class TriangleReprojRegressor(ReprojRegressor):
    def __init__(self, P1, P2, initial_params=np.array([1000, 1000, 500, 500, 1000, 1000, 500, 500, 1000, 1000, 500, 500,
                                              0, 0, 0, 0, 0, 0]),
                 bounds=np.array([[0, 3000], [0, 3000], [0, 2000], [0, 2000],
                                  [-10, 10], [-10, 10], [-10, 10],
                                  [-10, 10], [-10, 10], [-10, 10]]), verbose=1, method='lm'):
        self.P1 = P1
        self.P2 = P2
        super().__init__(initial_params, bounds, verbose, method)

    def fit(self, points12, points2d, sample_weight=None, nfev=None):
        result = least_squares(triangle_reproj_func, self.initial_params, args=(points2d, points12[:, :2], points12[:, 2:], self.P1, self.P2), verbose=self.verbose,
                               method=self.method, max_nfev=nfev)
        self.coef_ = result.x
        self.success = result.success
        return self

    def predict(self, points12):
        points3d = triangle_func(self.coef_[:8], points12[:, :2].reshape(-1, 1, 2), points12[:, 2:].reshape(-1, 1, 2), self.P1, self.P2)
        return proj_3dto2d(self.coef_[8:], points3d)


def DLS_all_pose_est(points12, points2d, P1, P2,
                     initial_params=np.array([1000, 1000, 500, 500, 1000, 1000, 500, 500, 1000, 1000, 500, 500,
                                              0, 0, 0, 0, 0, 0]), threshold=None, useRANSAC=True, max_trials=25,
                     bounds=np.array([[0, 3000], [0, 3000], [0, 2000], [0, 2000],
                                      [-10, 10], [-10, 10], [-10, 10],
                                      [-10, 10], [-10, 10], [-10, 10]])):
    '''
    threshold = 8.0 when not useFilter
              = adaptive when useFilter
    '''
    if useRANSAC:
        reg = RANSACRegressor(base_estimator=TriangleReprojRegressor(P1, P2, initial_params, bounds, verbose=1, method='lm'),
                              min_samples=max(np.ceil(len(points2d) * 0.01), 20),
                              loss=euclidean_err, residual_threshold=threshold,
                              is_model_valid=ReprojRegressor.is_model_valid, random_state=0, max_trials=max_trials)
        reg = reg.fit(points12, points2d)
        # if not reg.estimator_.success:
        #     reg.estimator_ = reg.estimator_.fit(points3d[reg.inlier_mask_], points2d[reg.inlier_mask_], nfev=50000)
        fx1, fy1, cx1, cy1, fx2, fy2, cx2, cy2, fx, fy, cx, cy, rx, ry, rz, tx, ty, tz = reg.estimator_.coef_
        R = np.array([[rx], [ry], [rz]])
        T = np.array([[tx], [ty], [tz]])
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        K1 = np.array([[fx1, 0, cx1], [0, fy1, cy1], [0, 0, 1]])
        K2 = np.array([[fx2, 0, cx2], [0, fy2, cy2], [0, 0, 1]])
        isPos = fx > 0 and fy > 0 and cx > 0 and cy > 0 and fx1 > 0 and fy1 > 0 and cx1 > 0 and cy1 > 0 and fx2 > 0 and fy2 > 0 and cx2 > 0 and cy2 > 0
        return reg.estimator_.success and isPos, R, T, K, reg.inlier_mask_, K1, K2
        # return reproject_avg_err(reg.estimator_.coef_, points2d[reg.inlier_mask_], points3d[reg.inlier_mask_]) < 8., R, T, K, reg.inlier_mask_
    else:
        # 进行非线性最小二乘优化
        result = least_squares(triangle_reproj_func, initial_params,
                               args=(points2d, points12[:, :2], points12[:, 2:], P1, P2),
                               verbose=1,
                               method='lm')
        test = result.fun < 10
        ratio = 1. * np.sum(test) / len(test)
        print(f'valid ratio:{ratio}')
        fx1, fy1, cx1, cy1, fx2, fy2, cx2, cy2, fx, fy, cx, cy, rx, ry, rz, tx, ty, tz = result.x
        R = np.array([[rx], [ry], [rz]])
        T = np.array([[tx], [ty], [tz]])
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        K1 = np.array([[fx1, 0, cx1], [0, fy1, cy1], [0, 0, 1]])
        K2 = np.array([[fx2, 0, cx2], [0, fy2, cy2], [0, 0, 1]])
        isPos = fx > 0 and fy > 0 and cx > 0 and cy > 0 and fx1 > 0 and fy1 > 0 and cx1 > 0 and cy1 > 0 and fx2 > 0 and fy2 > 0 and cx2 > 0 and cy2 > 0
        return result.success and isPos, R, T, K, K1, K2


def getInliners(kp1, kp2, matches, K1, K2, distCoeffs1, distCoeffs2, threshold, prob, no_intrinsic=False):
    if isinstance(kp1, cv2.KeyPoint):
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
    else:
        pts1 = np.float32([kp1[m[0]] for m in matches]).reshape(-1, 2)
        pts2 = np.float32([kp2[m[1]] for m in matches]).reshape(-1, 2)
    _, mask_F = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, threshold, prob)
    _, mask_E = cv2.findEssentialMat(pts1, pts2, K1, cv2.RANSAC, prob, threshold) if not no_intrinsic else (
    None, np.zeros(len(matches)))
    _, mask_H = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransacReprojThreshold=threshold, confidence=prob)

    inliners_F = np.sum(mask_F)
    inliners_E = np.sum(mask_E)
    inliners_H = np.sum(mask_H)

    if inliners_F >= inliners_E and inliners_F > inliners_H:
        inliner_num = inliners_F
        best_mask = mask_F
    elif inliners_E > inliners_H:
        inliner_num = inliners_E
        best_mask = mask_E
    else:
        inliner_num = inliners_H
        best_mask = mask_H

    inliners_idx = best_mask.ravel() == 1
    res = [matches[i] for i in range(len(matches)) if inliners_idx[i]]
    return res, inliner_num


def calcPose123(image1name, image2name, image3name, K1, K2, K3, P1, P2, match_path=None, distCoeffs=None, filter=True,
                threshold=3, prob=0.99):
    # Assuming image1, image2, and image3 are your input images
    image1 = cv2.imread(image1name, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2name, cv2.IMREAD_GRAYSCALE)
    image3 = cv2.imread(image3name, cv2.IMREAD_GRAYSCALE)
    # image1 = cv2.imread(image1, cv2.IMREAD_COLOR)
    # image2 = cv2.imread(image2, cv2.IMREAD_COLOR)
    # image3 = cv2.imread(image3, cv2.IMREAD_COLOR)

    # Create SIFT detector
    # sift = cv2.SIFT_create()
    sift = cv2.ORB_create()

    # Find the keypoints and descriptors with SIFT
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)
    keypoints3, descriptors3 = sift.detectAndCompute(image3, None)
    if descriptors1 is None or descriptors2 is None or descriptors3 is None:
        print("Empty descriptors, cannot build FLANN index")
        return None
    descriptors1 = descriptors1.astype(np.float32)
    descriptors2 = descriptors2.astype(np.float32)
    descriptors3 = descriptors3.astype(np.float32)

    # FLANN parameters
    # FLANN_INDEX_KDTREE = 1
    # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    # search_params = dict(checks=100)

    # FLANN Matcher
    # flann = cv2.FlannBasedMatcher(index_params, search_params)
    # distant_type = cv2.NORM_HAMMING
    # distant_type = cv2.NORM_L2
    bf = cv2.BFMatcher(crossCheck=True)
    # bf = cv2.BFMatcher(crossCheck=True)

    # Match descriptors using FLANN
    assert descriptors1.shape[1] == descriptors2.shape[1], "Descriptor sizes do not match"

    if descriptors1.size == 0 or descriptors2.size == 0 or descriptors3.size == 0:
        print("Empty descriptors, cannot build FLANN index")
        return None
    # matches13 = flann.knnMatch(descriptors1, descriptors3, k=2)
    # matches23 = flann.knnMatch(descriptors2, descriptors3, k=2)

    matches13 = bf.match(descriptors1, descriptors3)
    matches23 = bf.match(descriptors2, descriptors3)
    good_matches13 = sorted(matches13, key=lambda x: x.distance)[:1000]
    good_matches23 = sorted(matches23, key=lambda x: x.distance)[:1000]

    # determine inliners
    if filter:
        m13, num13 = getInliners(keypoints1, keypoints3, good_matches13, K1, K3, distCoeffs, distCoeffs, threshold,
                                 prob)
        m23, num23 = getInliners(keypoints2, keypoints3, good_matches23, K2, K3, distCoeffs, distCoeffs, threshold,
                                 prob)
        if num13 > 60 and num23 > 60:
            good_matches13 = m13
            good_matches23 = m23

    # ratio = 0.75
    # good_matches13 = set()
    # for m, n in matches13:
    #     if m.distance < ratio * n.distance:
    #         good_matches13.add(m)
    print(f"matches13 num:{len(good_matches13)}")
    #
    # good_matches23 = set()
    # for m, n in matches23:
    #     if m.distance < ratio * n.distance:
    #         good_matches23.add(m)
    print(f"matches23 num:{len(good_matches23)}")

    # orb = cv2.ORB_create()
    # keypoints1, descriptions1 = orb.detectAndCompute(image1, None)
    # keypoints2, descriptions2 = orb.detectAndCompute(image2, None)
    # keypoints3, descriptions3 = orb.detectAndCompute(image3, None)

    # Convert descriptors to np.float32
    # descriptions1 = descriptions1.astype(np.float32)
    # descriptions2 = descriptions2.astype(np.float32)
    # descriptions3 = descriptions3.astype(np.float32)

    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # matches13 = bf.match(descriptions1, descriptions3)
    # matches32 = bf.match(descriptions3, descriptions2)
    # matches13 = sorted(matches13, key=lambda x: x.distance)[:1000]
    # matches32 = sorted(matches32, key=lambda x: x.distance)[:1000]
    # print(matches13)
    # print(matches32)
    # matches31 = {m.trainIdx: m.queryIdx for m in matches13}
    # matches32 = {m.queryIdx: m.trainIdx for m in matches32}
    matches31 = {}
    for m in good_matches13:
        if m.trainIdx not in matches31:
            matches31[m.trainIdx] = m.queryIdx
    matches32 = {}
    for m in good_matches23:
        if m.trainIdx not in matches32:
            matches32[m.trainIdx] = m.queryIdx
    # matches31 = {m.trainIdx: m.queryIdx for m in good_matches13}
    # matches32 = {m.trainIdx: m.queryIdx for m in good_matches23}
    print(f"matches31 num:{len(matches31)}")
    print(f"matches32 num:{len(matches32)}")
    common_points_idx = set(matches31.keys()).intersection(set(matches32.keys()))
    # print(common_points_idx)
    points3 = np.float32([keypoints3[i].pt for i in common_points_idx]).reshape(-1, 1, 2)
    print(f"matches123 num:{points3.shape[0]}")
    if points3.shape[0] == 0:
        print('pose est failed!')
        return None
    if match_path:
        matches123 = [cv2.DMatch(matches32[i], i, 0) for i in common_points_idx]
        img_with_key = cv2.drawMatches(image2, keypoints2, image3, keypoints3, matches123, None)
        # cv2.namedWindow('keypoints', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('keypoints', min(img_with_key.shape[1], 1920-100), min(img_with_key.shape[0], 1080-100))
        compression_params = [int(cv2.IMWRITE_JPEG_QUALITY), 60]
        cv2.imwrite(os.path.join(match_path,
                                 'match_' + os.path.basename(image2name).split('.')[0] + os.path.basename(image3name)),
                    img_with_key, compression_params)

    points1 = np.float32([keypoints1[matches31[i]].pt for i in common_points_idx]).reshape(-1, 1, 2)
    points2 = np.float32([keypoints2[matches32[i]].pt for i in common_points_idx]).reshape(-1, 1, 2)
    # points1 = pixel2cam(points1, K1).reshape(-1, 1, 2)
    # points2 = pixel2cam(points2, K2).reshape(-1, 1, 2)
    points3d = cv2.triangulatePoints(np.dot(K1, P1), np.dot(K2, P2), points1, points2)
    points3d = cv2.convertPointsFromHomogeneous(points3d.T).squeeze()

    if points3d.shape[0] > 40:
        success, R, T, inliners = cv2.solvePnPRansac(points3d, points3, K3, distCoeffs)
    elif points3d.shape[0] >= 6:
        success, R, T = cv2.solvePnP(points3d, points3, K3, distCoeffs)
    else:
        success = False
    if success:
        R, _ = cv2.Rodrigues(R)
        pose = np.hstack((R, T))
        return pose
    else:
        print('pose est failed!')
        return None
