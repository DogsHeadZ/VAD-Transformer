import numpy as np
from collections import OrderedDict
import os
import glob
import random
import cv2
import torch
import torch.utils.data as data
from getBox import *
from getFlow import *
import torch.nn.functional as F

def np_load_frame_roi(filename, image_height, image_width, resize_height, resize_width, bbox):
    (xmin, ymin, xmax, ymax) = bbox
    image_decoded = cv2.imread(filename)
    image_decoded = cv2.resize(image_decoded, (image_height, image_width))

    image_decoded = image_decoded[ymin:ymax, xmin:xmax]

    image_resized = cv2.resize(image_decoded, (resize_width, resize_height))
    # image_resized = image_resized.astype(dtype=np.float32)
    # image_resized = (image_resized / 127.5) - 1.0
    return image_resized


def np_load_frame(filename, resize_height, resize_width):
    """
    Load image path and convert it to numpy.ndarray. Notes that the color channels are BGR and the color space
    is normalized from [0, 255] to [-1, 1].
    :param filename: the full path of image
    :param resize_height: resized height
    :param resize_width: resized width
    :return: numpy.ndarray
    """
    image_decoded = cv2.imread(filename)
    image_resized = cv2.resize(image_decoded, (resize_width, resize_height))
    # image_resized = image_resized.astype(dtype=np.float32)
    # image_resized = (image_resized )/255.0

    # image_resized = (image_resized / 127.5) - 1.0
    return image_resized


class VadDataset(data.Dataset):
    def __init__(self, fastrcnn_config, video_folder, transform, resize_height, resize_width, dataset='', time_step=4, num_pred=1,
                 bbox_folder=None, device='cuda:0', flow_folder=None):
        self.fastrcnn_config = fastrcnn_config
        self.dir = video_folder
        self.transform = transform
        self.videos = OrderedDict()
        self._resize_height = resize_height
        self._resize_width = resize_width
        self._time_step = time_step
        self._num_pred = num_pred
        self.dataset = dataset  # ped2 or avenue or ShanghaiTech

        self.bbox_folder = bbox_folder  # 如果box已经预处理了，则直接将npy数据读出来, 如果没有，则在get_item的时候计算
        if bbox_folder == None:  # 装载yolo模型

            self.model = COCODemo(
                self.fastrcnn_config,
                min_image_size=800,
                confidence_threshold=0.5,
            )

        self.setup()
        self.samples = self.get_all_samples()

    def setup(self):
        videos = glob.glob(os.path.join(self.dir, '*'))
        for video in sorted(videos):
            # video_name = video.split('/')[-1]           #视频的目录名即类别如01, 02, 03, ...
            video_name = os.path.split(video)[-1]
            self.videos[video_name] = {}
            self.videos[video_name]['path'] = video
            self.videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))  # 每个目录下的所有视频帧
            self.videos[video_name]['frame'].sort()
            self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])  # 每个目录下视频帧的个数

        video_name = os.path.split(videos[0])[-1]
        self.img_size = cv2.imread(self.videos[video_name]['frame'][0]).shape  # [h, w, c]
        if self.bbox_folder != None:  # 如果box已经预处理了，则直接将npy数据读出来
            for bbox_file in sorted(os.listdir(self.bbox_folder)):
                video_name = bbox_file.split('.')[0]
                self.videos[video_name]['bbox'] = np.load(os.path.join(self.bbox_folder, bbox_file),
                                                          allow_pickle=True)  # 每个目录下所有视频帧预提取的bbox

    def get_all_samples(self):
        frames = []
        videos = glob.glob(os.path.join(self.dir, '*'))
        for video in sorted(videos):
            # video_name = video.split('/')[-1]
            video_name = os.path.split(video)[-1]
            for i in range(len(self.videos[video_name]['frame']) - self._time_step):  # 减掉_time_step为了刚好能够滑窗到视频尾部
                frames.append(self.videos[video_name]['frame'][i])  # frames存储着训练时每段视频片段的首帧，根据首帧向后进行滑窗即可得到这段视频片段

        return frames

    def __getitem__(self, index):
        # video_name = self.samples[index].split('/')[-2]      #self.samples[index]取到本次迭代取到的视频首帧，根据首帧能够得到其所属类别及图片名
        # frame_name = int(self.samples[index].split('/')[-1].split('.')[-2])

        # windows
        video_name = os.path.split(os.path.split(self.samples[index])[0])[1]
        frame_name = int(os.path.split(self.samples[index])[1].split('.')[-2])

        if self.bbox_folder is not None:  # 已经提取好了bbox，直接读出来
            # bboxes = self.videos[video_name]['bbox'][frame_name + self._time_step - 1]
            bboxes = self.videos[video_name]['bbox'][frame_name + self._time_step // 2] # 取中间那帧的box
        else:  # 需要重新计算
            all_bboxes = []
            all_feats = []

            for i in range(self._time_step):
                frame = self.videos[video_name]['frame'][frame_name + i]
                bboxes, feats = RoIResize(frame, self.dataset, self.model,self._resize_height,
                                  self._resize_width)
                all_bboxes.append(bboxes)
                all_feats.append(feats.squeeze(0))

            all_feats = torch.stack(all_feats, dim=0)

        middleboxes = np.array(all_bboxes[self._time_step // 2])
        # w_ratio = self._resize_width / self.img_size[1]
        # h_ratio = self._resize_height / self.img_size[0]
        # trans_bboxes = [[int(box[0] * w_ratio), int(box[1] * h_ratio),
        #                  int(box[2] * w_ratio), int(box[3] * h_ratio)] for box in middleboxes]

        # frame_batch = []
        # for i in range(self._time_step + self._num_pred):
        #     image = np_load_frame(self.videos[video_name]['frame'][frame_name + i], self._resize_height,
        #                           self._resize_width)  # 根据首帧图片名便可加载一段视频片段
        #     if self.transform is not None:
        #         frame_batch.append(self.transform(image))
        # frame_batch = torch.stack(frame_batch, dim=0)  # 大小为[_time_step+num_pred, c, _resize_height, _resize_width]

        if middleboxes.shape[0] == 0:
            object_batch = np.array([])
        else:
            object_batch = []
            for i in range(self._time_step + self._num_pred):
                one_object_batch = []
                for bbox in middleboxes:
                    image = np_load_frame_roi(self.videos[video_name]['frame'][frame_name + i], self._resize_height,
                                      self._resize_width, 64, 64, bbox)  # 这里的64是裁剪框的resize
                    if self.transform is not None:
                        one_object_batch.append(self.transform(image))
                one_object_batch = torch.stack(one_object_batch, dim=0)
                # print("object_batch.shape: ", object_batch.shape)
                object_batch.append(one_object_batch)

            object_batch = torch.stack(object_batch,
                                       dim=0)  # 大小为[_time_step+num_pred, 目标个数, 图片的通道数, _resize_height, _resize_width]

        return all_feats, object_batch, middleboxes
    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    # test dataloader
    import torchvision
    from torchvision import datasets
    from torch.utils.data import Dataset, DataLoader
    import matplotlib.pyplot as plt
    import torchvision.transforms as transforms

    parser = argparse.ArgumentParser()

    # for flownet2, no need to modify
    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument("--rgb_max", type=float, default=255.)
    args = parser.parse_args()

    batch_size = 1
    datadir = "../../AllDatasets/avenue/training/frames"
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


    # 使用保存的.npy
    test_data = VadDataset(args,video_folder=datadir, bbox_folder = None, dataset='avenue', flow_folder=None,
                            transform=transforms.Compose([transforms.ToTensor()]),
                            resize_height=256, resize_width=256)


    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    unloader = transforms.ToPILImage()

    X, feats, X_bbox, bboxes = next(iter(test_loader))
    print(X.shape)
    print(X_bbox.shape)
    print(feats.shape)

    def plot_one_box(x, img, color=None, label=None, line_thickness=None):
        # Plots one bounding box on image img
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


    # 显示一个batch

    index = 1
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            plt.subplot(X.shape[0], X.shape[1], index)
            index += 1

            img = X[i,j,:,:,:].cpu().clone()

            img = unloader(img)
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            for bbox in bboxes:
                bbox = [int(a1) for a1 in bbox]
                plot_one_box(bbox, img)
            # cv2.imwrite('roi.jpg', img)
            plt.imshow(img)

    plt.savefig('frame.jpg')

    plt.figure()
    index = 1
    X_bbox = X_bbox.squeeze(0)
    for i in range(X_bbox.shape[0]):
        for j in range(X_bbox.shape[1]):
            plt.subplot(X_bbox.shape[0], X_bbox.shape[1], index)
            index += 1

            img = X_bbox[i, j, :, :, :].cpu().clone()

            img = unloader(img)
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img)

    plt.savefig('object.jpg')