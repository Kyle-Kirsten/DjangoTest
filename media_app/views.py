# media_app/views.py
import json

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage
# from django.conf import settings
from django_project import settings
from .models import Image, Video
from .forms import VideoForm, ImageForm
from utils.upload import new_name, new_dir_name
from utils.calib3d import *
from datetime import datetime
import os, re
import subprocess
import numpy as np
import cv2, numpy
import time

from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet
from lightglue.utils import load_image, rbd, read_image, numpy_image_to_torch, resize_image
from lightglue import viz2d
import torch


def upload_video(request):
    if request.method == 'POST':
        custom_location = request.GET.get('custom_location', 'videos/')
        isRename = request.GET.get('isRename', 'false').lower()
        isRename = isRename == 'true'
        form = VideoForm(request.POST, request.FILES)
        if form.is_valid():
            instance = form.save(commit=False)
            instance.video.name = new_name(custom_location,
                                           instance.video.name, isRename)
            # Assign the custom location to the image instance
            res = instance.save()
            return JsonResponse({'message': 'Video uploaded successfully', 'saved_path': res},
                                status=200)
    else:
        form = VideoForm()
    return render(request, 'upload_video.html', {'form': form})


@csrf_exempt
def upload_image(request):
    if request.method == 'POST':
        custom_location = request.GET.get('custom_location', 'images/')
        print(custom_location)
        isRename = request.GET.get('isRename', 'false').lower()
        isRename = isRename == 'true'
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            image_instance = form.save(commit=False)
            image_instance.image.name = new_name(custom_location,
                                                 image_instance.image.name, isRename)
            # Assign the custom location to the image instance
            res = image_instance.save()
            return JsonResponse({'message': 'Image uploaded successfully', 'saved_path': res},
                                status=200)
        else:
            image = request.FILES['image']
            image_instance = Image(image=image)
            image_instance.image.name = new_name(custom_location, image.name)
            res = image_instance.save()
            return JsonResponse({'err': 'form invalid', 'saved_path': res}, status=404)
    else:
        form = ImageForm()
    return render(request, 'upload_image.html', {'form': form})


@csrf_exempt
def upload_multiple_images(request):
    if request.method == 'POST' and request.FILES.getlist('images'):
        custom_location = request.GET.get('custom_location', 'images/')
        isRename = request.GET.get('isRename', 'false').lower()
        isRename = isRename == 'true'
        res = []
        for image_file in request.FILES.getlist('images'):
            image_instance = Image(image=image_file)
            image_instance.image.name = new_name(custom_location,
                                                 image_instance.image.name, isRename)
            res.append(image_instance.save())
        return JsonResponse(
            {'message': 'Images uploaded successfully to custom location', 'saved_path': res},
            status=200)
    return JsonResponse({'error': 'POST request and images required'},
                        status=400)


def upload_multiple_videos(request):
    if request.method == 'POST' and request.FILES.getlist('videos'):
        custom_location = request.GET.get('custom_location', 'videos/')
        isRename = request.GET.get('isRename', 'false').lower()
        isRename = isRename == 'true'
        res = []
        for video_file in request.FILES.getlist('videos'):
            instance = Video(video=video_file)
            instance.video.name = new_name(custom_location,
                                           instance.video.name, isRename)
            res.append(instance.save())
        return JsonResponse({'message': 'Videos uploaded successfully', 'saved_path': res},
                            status=200)
    return JsonResponse({'error': 'POST request and videos required'},
                        status=400)


@csrf_exempt
def request_colmap_auto(request):
    if request.method == 'GET':
        request_location = request.GET.get('request_location', 'temps/')
        save_location = request.GET.get('save_location', 'temps/')
        sav_loc = os.path.join(settings.MEDIA_ROOT, 'colmaps/', save_location)
        if not os.path.exists(sav_loc):
            os.makedirs(sav_loc)
        colmap_params = request.GET.get('colmap_params', 'automatic_reconstructor')
        colmap_params = colmap_params + ' ' + request.POST.get('colmap_params', '')
        folder = os.path.join(settings.MEDIA_ROOT, 'images/', request_location)
        if os.path.exists(folder):
            command = settings.COLMAP_PATH + ' ' + colmap_params + ' --image_path ' + \
                      os.path.join(settings.MEDIA_ROOT, 'images/', request_location) + \
                      ' --workspace_path ' + \
                      os.path.join(settings.MEDIA_ROOT, 'colmaps/', save_location)
            subprocess.run(command, shell=True)
            return JsonResponse({'messag e': 'Folder found', 'saved_path': sav_loc}, status=200)
        else:
            return JsonResponse(
                {'error': 'No images found in the specified folder'},
                status=404)

    return JsonResponse({'error': 'Get request required'}, status=400)


@csrf_exempt
def request_colmap(request):
    if request.method == 'GET':
        project_location = request.GET.get('project_location', 'temps/')
        sav_loc = os.path.join(settings.MEDIA_ROOT, 'colmaps/', project_location)
        if not os.path.exists(sav_loc):
            os.makedirs(sav_loc)
        colmap_params = request.GET.get('colmap_params', '')
        colmap_params = colmap_params + ' ' + request.POST.get('colmap_params', '')
        command = 'cd ' + sav_loc + ' && ' + settings.COLMAP_PATH + ' ' + colmap_params
        subprocess.run(command, shell=True)
        return JsonResponse({'messag e': 'Folder found', 'saved_path': sav_loc}, status=200)

    return JsonResponse({'error': 'Get request required'}, status=400)


@csrf_exempt
def request_NVLAD(request):
    req_loc = request.GET.get('request_location', 'temps/')
    req_loc = os.path.join(req_loc, 'color')
    if req_loc[-1] != '/':
        req_loc = req_loc + '/'
    save_loc = os.path.join(settings.MEDIA_ROOT, 'nvlabs/', request.GET.get('save_location', req_loc))
    if save_loc[-1] != '/':
        save_loc = save_loc + '/'
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)
        os.makedirs(os.path.join(save_loc, 'index_features/'))
    netpath = settings.NetVLAD_PATH
    nv_params = request.GET.get('nv_params', '')
    folder = os.path.join(settings.MEDIA_ROOT, 'images/', req_loc)
    if os.path.exists(folder):
        # feature extraction
        # command_conda = "source /home/vr717/anaconda3/etc/profile.d/conda.sh && conda activate patchnetvlad "
        # command_conda = command_conda + f'&& bash make_dataset_and_extract.sh {os.path.join(settings.MEDIA_ROOT, "images/", req_loc)} {save_loc} '
        # command = f'cd {netpath} && bash -c "{command_conda}"'
        command = f'cd {netpath} && bash make_dataset_and_extract.sh {os.path.join(settings.MEDIA_ROOT, "images/", req_loc)} {save_loc} '
        # subprocess.run(command, shell=True)
        os.system(command)
        return JsonResponse({'message': 'Folder Found', 'saved_path': save_loc}, status=200)
    else:
        return JsonResponse({'error': 'NO such folder'}, status=404)


def image_transform(image: numpy, use_equalizeHist=True):
    (height, width) = image.shape[:2]
    if height >= width:
        return image
    # rotate the src image 90 degrees clockwise
    # rotated_image = cv2.transpose(image)
    rotated_image = cv2.flip(image, -1)
    rotated_image = cv2.flip(rotated_image, 1) # used when the image is not mirrored
    if use_equalizeHist:
        rotated_image = cv2.equalizeHist(rotated_image)
    return rotated_image


@csrf_exempt
def request_NVLAD_redir(request):
    start_init = time.time()
    img_loc = request.GET.get('source_location', 'temps/')
    img_loc = os.path.join(img_loc, 'color')
    src_loc = os.path.join(settings.MEDIA_ROOT, 'images/', img_loc)
    dataset_loc = img_loc.split('/')[0]
    intri_loc = os.path.join(settings.MEDIA_ROOT, 'images/',
                             request.GET.get('camera_intri_location', os.path.join(dataset_loc, 'intrinsic')))
    exter_loc = os.path.join(settings.MEDIA_ROOT, 'images/',
                             request.GET.get('camera_exter_location', os.path.join(dataset_loc, 'pose')))
    req_loc = os.path.join(settings.MEDIA_ROOT, 'nvlabs/', request.GET.get('request_location', img_loc))
    if req_loc[-1] != '/':
        req_loc = req_loc + '/'
    if src_loc[-1] != '/':
        src_loc = src_loc + '/'

    if not os.path.exists(req_loc) or not os.path.exists(src_loc):
        return JsonResponse({'error': 'No such folder'}, status=404)
    nv_params = request.GET.get('nv_params', '')
    if request.method == 'POST':
        tempfolder = os.path.join(req_loc, new_dir_name('query'))
        tempfeature = os.path.join(tempfolder, 'query_features')
        tempimages = os.path.join(tempfolder, 'query_folder')
        tempquery = os.path.join(tempfolder, 'query.txt')
        print(req_loc)
        print(tempfolder)
        print(tempquery)
        if not os.path.exists(tempfolder):
            os.makedirs(tempfolder)
            os.makedirs(tempfeature)
            os.makedirs(tempimages)

        images = request.FILES.getlist('images')

        if images is None or len(images) == 0:
            print('single image')
            images = [request.FILES.get('images')]
        if not any(images):
            return JsonResponse({'error': 'post image required'}, status=404)
        storage_path = tempimages

        saved_images = []
        K = np.array([[1428.643433, 0.000000, 1428.643433],
                      [0.000000, 970.724121, 716.204285],
                      [0.000000, 0.000000, 1.000000]])
        camera_matrix = request.POST.get('camera_matrix')
        if camera_matrix is not None:
            camera_matrix = json.loads(camera_matrix)
        qintrinsic = {}

        with open(tempquery, 'w') as qtxt:
            for i, image in enumerate(images):
                image_name, _ = os.path.splitext(image.name)
                image_name = image_name + '.jpg'
                if camera_matrix is not None and i < len(camera_matrix):
                    qintrinsic[image_name] = np.array(camera_matrix[i])
                # qintrinsic[image.name] = np.array(camera_matrix[i]) if camera_matrix and i < len(camera_matrix) else K
                fs = FileSystemStorage(location=storage_path)
                saved_image = fs.save(image_name, image)
                saved_images.append(
                    save_to_jpg(os.path.join(storage_path, saved_image), os.path.join(storage_path, image_name)))
                qtxt.write(image_name + '\n')


        command = f'cd {settings.NetVLAD_PATH} && bash match_and_cal_pose.sh {req_loc} {src_loc} {tempfolder}'

        os.system(command)

        start = time.time()
        resfolder = os.path.join(tempfolder, 'result/')
        positions = {}

        pred_pattern = re.compile(r',\s*')
        pred_imgs = {}
        max_pred_num = 3

        if not os.path.exists(os.path.join(resfolder, 'PatchNetVLAD_predictions.txt')):
            return JsonResponse({'error': 'PatchNetVLAD failed to match'}, status=200)
        with open(os.path.join(resfolder, 'PatchNetVLAD_predictions.txt'), 'r') as qtxt:
            for i, line in enumerate(qtxt, start=1):
                if not line.startswith('#'):
                    ims = re.split(pred_pattern, line.strip())
                    ims[0] = ims[0].strip()
                    ims[1] = ims[1].strip()
                    _, qimname = os.path.split(ims[0])
                    _, simname = os.path.split(ims[1])
                    if qimname not in pred_imgs.keys():
                        pred_imgs[qimname] = [
                            (ims[1], os.path.join(intri_loc, simname.split('.')[0] + '.intrinsic_color.txt'),
                             os.path.join(exter_loc, simname.split('.')[0] + '.pose.txt'))]
                    elif len(pred_imgs[qimname]) < max_pred_num:
                        pred_imgs[qimname].append(
                            (ims[1], os.path.join(intri_loc, simname.split('.')[0] + '.intrinsic_color.txt'),
                             os.path.join(exter_loc, simname.split('.')[0] + '.pose.txt')))

        feature_extractor = SuperPoint(max_num_keypoints=2048).eval().to(settings.DEVICE)  # load the extractor
        feature_match = LightGlue(features="superpoint").eval().to(settings.DEVICE)
        match_num = 500
        use_gray = True
        use_default_pose = True
        distCoeffs = None
        useFilter = False
        filter_num = 100
        filter_params = {'distCoeffs1': None, 'distCoeffs2': None, 'threshold': 4., 'prob': 0.99, 'no_intrinsic': True}
        drawMatch = True
        timeout = 45
        W = 640
        H = 480
        est_focal = np.sqrt(W ** 2 + H ** 2) * 1428.643433 / 1440
        est_K = np.array([[est_focal, 0, W / 2.], [0, est_focal, H / 2.], [0, 0, 1]])
        for qimname, v in pred_imgs.items():
            qim = os.path.join(tempimages, qimname)
            image3 = read_image(qim, grayscale=use_gray)
            print("image3 src shape is :", image3.shape)
            image3 = image_transform(image3)
            image3, _ = resize_image(image3, (H, W))
            # image3 = cv2.resize(image3, (W, H))
            cv2.imwrite(os.path.join(resfolder, "query.jpg"), image3)
            K3 = est_K
            print(f'image3 intrinsic:{K3}')
            print(f'image3 shape:{image3.shape}')

            default_P = read_pose_3dscanner(v[0][2]) if os.path.exists(v[0][2]) else np.eye(3, 4)
            ground_truth = os.path.join(exter_loc, qimname.split('.')[0] + '.pose.txt')
            ground_P3 = read_pose_3dscanner(ground_truth) if os.path.exists(ground_truth) else default_P

            if use_default_pose:
                print("use default pose")
                pose = default_P
                positions[qimname] = pose.tolist()
                end = time.time()
                print(f'pnp time cost:{end - start}')
                print(f'total time cost:{end - start_init}')
                continue

            points3_all = np.zeros((0, 1, 2), dtype=np.float)
            points3d_all = np.zeros((0, 3), dtype=np.float)

            for i in range(0, len(v) - 1):
                sim1 = v[i][0]
                image1 = read_image(sim1, grayscale=use_gray)
                print(f'image1 shape:{image1.shape}')
                # image1 = image_transform(image1)
                image1, _ = resize_image(image1, (H, W))
                # image1 = cv2.resize(image1, (W, H))
                if i == 0:
                    cv2.imwrite(os.path.join(resfolder, "default.jpg"), image1)
                K1 = est_K
                print(f'image1 intrinsic:{K1}')

                P1 = read_pose_3dscanner(v[i][2]) if os.path.exists(v[i][2]) else np.eye(3, 4)
                print(f'image1 pose:{P1}')

                feats1 = feature_extractor.extract(numpy_image_to_torch(image1).to(settings.DEVICE))
                feats1out = rbd(feats1)
                kp1 = feats1out['keypoints'].cpu().numpy()
                feats3 = feature_extractor.extract(numpy_image_to_torch(image3).to(settings.DEVICE))
                feats3out = rbd(feats3)
                kp3 = feats3out['keypoints'].cpu().numpy()
                matches13 = feature_match({"image0": feats1, "image1": feats3})
                matches13out = rbd(matches13)
                scores, sort_idx = torch.sort(matches13out['scores'], descending=True)
                good_matches13 = matches13out['matches'][sort_idx[:match_num]].cpu().numpy()
                if useFilter:
                    m13, num13 = getInliners(kp1, kp3, good_matches13, K1, K3, **filter_params)
                    if num13 > filter_num:
                        good_matches13 = np.array(m13)

                for j in range(i + 1, len(v)):
                    sim2 = v[j][0]
                    image2 = read_image(sim2, grayscale=use_gray)
                    # image2 = image_transform(image2)
                    image2, _ = resize_image(image2, (H, W))
                    # image2 = cv2.resize(image2, (W, H))
                    K2 = est_K
                    P2 = read_pose_3dscanner(v[j][2]) if os.path.exists(v[j][2]) else np.eye(3, 4)
                    feats2 = feature_extractor.extract(numpy_image_to_torch(image2).to(settings.DEVICE))
                    feats2out = rbd(feats2)
                    kp2 = feats2out['keypoints'].cpu().numpy()
                    matches12 = feature_match({"image0": feats1, "image1": feats2})
                    matches12out = rbd(matches12)
                    scores, sort_idx = torch.sort(matches12out['scores'], descending=True)
                    good_matches12 = matches12out['matches'][sort_idx[:match_num]].cpu().numpy()
                    if useFilter:
                        m12, num12 = getInliners(kp1, kp2, good_matches12, K1, K2, **filter_params)
                        if num12 > filter_num:
                            good_matches12 = np.array(m12)
                    matches123 = np.zeros((0, 3), dtype=np.long)
                    # print(good_matches12)
                    # print(good_matches13)
                    for m13 in good_matches13:
                        mask = m13[0] == good_matches12[:, 0]
                        if mask.any():
                            # print(mask)
                            # print(matches123)
                            m123 = np.concatenate((good_matches12[mask][0], m13[1:])).reshape(1, 3)
                            matches123 = np.concatenate((matches123, m123))
                    if matches123.size < 10:
                        print("common points too low, pose est failed!")
                        continue
                    points1 = (kp1[matches123[:, 0]]).reshape(-1, 1, 2)
                    points2 = (kp2[matches123[:, 1]]).reshape(-1, 1, 2)
                    points3_all = np.concatenate((points3_all, (kp3[matches123[:, 2]]).reshape(-1, 1, 2)))
                    # points1 = np.copy(kp1[matches123[:, 0]]).reshape(-1, 1, 2)
                    # points2 = np.copy(kp2[matches123[:, 1]]).reshape(-1, 1, 2)
                    # points3 = np.copy(kp3[matches123[:, 2]]).reshape(-1, 1, 2)
                    points3d = cv2.triangulatePoints(K1 @ P1, K2 @ P2, points1, points2)
                    points3d = cv2.convertPointsFromHomogeneous(points3d.T).squeeze()
                    points3d_all = np.concatenate((points3d_all, points3d.reshape(-1, 3)))
                    torch.cuda.empty_cache()

            rvec, _ = cv2.Rodrigues(default_P[:3, :3])
            tvec = default_P[:, 3:]
            inliners = np.arange(len(points3_all))
            success, R, T, inliners = cv2.solvePnPRansac(points3d_all, points3_all, K3, distCoeffs, useExtrinsicGuess=False, rvec=rvec, tvec=tvec)
            # success, R, T, K, inliners = DLS_pose_est(points3d_all.reshape(-1, 3), points3_all.reshape(-1, 2),
            #                                           initial_params=pose2params(est_K, default_P[:3, :3], default_P[:3, 3:]), max_trials=100)
            # success, R, T = cv2.solvePnP(points3d_all[inliners], points3_all[inliners], K3, distCoeffs,
            #                              useExtrinsicGuess=False, rvec=rvec, tvec=tvec)
            if success or inliners is not None:
                inliners = inliners.squeeze()
                print(f'inliner num:{inliners.shape}')
                if not success:
                    success, R, T = cv2.solvePnP(points3d_all[inliners], points3_all[inliners], K3, distCoeffs, useExtrinsicGuess=False, rvec=rvec, tvec=tvec)
                if success:
                    Rtmp, _ = cv2.Rodrigues(R)
                    pose = np.hstack((Rtmp, T))
                else:
                    print("pose est failed")
                    pose = default_P
            else:
                print("pose est failed")
                pose = default_P

            positions[qimname] = pose.tolist()

            end = time.time()
            print(f'pnp time cost:{end - start}')
            print(f'total time cost:{end - start_init}')

            if ground_P3 is not None and pose is not None:
                R3 = ground_P3[:3, :3]
                R3_qim = pose[:3, :3]
                residuals = ground_P3 - pose
                rot_vec_p3, _ = cv2.Rodrigues(R3)
                rot_vec_qim, _ = cv2.Rodrigues(R3_qim)
                print(f'Loss shift:{np.linalg.norm(residuals[:, 3:])}')
                print(f'Loss rot:{np.linalg.norm(residuals[:, :3])}')
                print(
                    f'Loss rot radius:{(np.linalg.norm(rot_vec_p3) - np.linalg.norm(rot_vec_qim)) * 180. / np.pi}')
                print(
                    f'Loss rot vec dir:{np.linalg.norm(rot_vec_p3 / np.linalg.norm(rot_vec_p3) - rot_vec_qim / np.linalg.norm(rot_vec_qim))}')

        return JsonResponse({'message': 'Folder Found', 'saved_path': saved_images, 'positions': positions}, status=200)

    else:
        return JsonResponse({'error': 'POST request required'}, status=400)
