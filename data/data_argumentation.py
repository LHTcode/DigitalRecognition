import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
from config.global_config import global_config
from collections import namedtuple
from typing import Tuple, List

class RandomNoise(T.RandomErasing):
    """Make some white point noise.
    This class inherits from torchvision.transforms.RandomErasing. The detail
    of parameters introduction see https://pytorch.org/vision/stable/generated/torchvision.transforms.RandomErasing.html?highlight=randomerasing#randomerasing
    'Random Erasing Data Augmentation' by Zhong et al. See https://arxiv.org/abs/1708.04896

    Args:
        p: probability that the random noise operation will be performed.
        scale: range of proportion of erased area against input image.
        ratio: range of aspect ratio of erased area.
        erasing_area: white point noise area, default is 100.

    Returns:
        Added white point noise Image.
    """
    def __init__(self, p=0.5, scale=(0.0006, 0.0006), ratio=(1, 1), erasing_area=100):
        super(RandomNoise, self).__init__(1, scale, ratio, 255)
        self.p = p
        self.erasing_area = erasing_area

    def forward(self, img):
        if torch.rand(1) < self.p:
            value = [self.value]
            for _ in range(self.erasing_area):
                if value is not None and not (len(value) in (1, img.shape[-3])):
                    raise ValueError(
                        "If value is a sequence, it should have either a single value or "
                        f"{img.shape[-3]} (number of input channels)"
                    )
                x, y, h, w, v = self.get_params(img, scale=self.scale, ratio=self.ratio, value=value)
                img = F.erase(img, x, y, h, w, v, self.inplace)
            return img
        return img

class DataAugmentation:
    CROP_AREA: float = 0.5
    SHEAR_DEGREES: int = 45
    WHITE_POINT_AREA: float = 0.0006
    METHODS_LIST: List[str] = ['random_resized_crop', 'random_affine', 'random_noise']

    """Provide some data augmentation method.

    Note: You should give names of some data argumentation methods name,
    and use get_data_argumentation_compose method to get a transform compose.

    Args:
        methods (list of str): except some data argumentation methods name.
    """
    def __init__(self, methods: List[str]):
        self.methods = methods
        self.Shape = namedtuple(typename="img_shape", field_names=['H', 'W'])
        for idx, method in enumerate(self.methods):
            if method not in self.METHODS_LIST:
                print(f"Warning: method named {method} is no found!")
                del self.methods[idx]

    def get_data_argumentation_compose(self, img_shape: Tuple[int, int]) -> T.Compose:
        """Get some data argumentation methods by torchvision.transform.Compose format.

        Effects test:
        >>> import cv2
        >>> dataset_path = global_config.DATASET_PATH
        >>> src_img = cv2.imread(dataset_path + "/train" + '/2' + '/252.png', cv2.IMREAD_GRAYSCALE)
        >>> src_img = torch.from_numpy(src_img)
        >>> data_arug = DataAugmentation(['random_resized_crop', 'random_affine', 'random_noise'])
        >>> transform = data_arug.get_data_argumentation_compose(src_img.shape)
        >>> for i in range(10):
        ...     random_resized_crop_img = transform(src_img.unsqueeze(0))
        ...     cv2.imshow("random_resized_crop_img", random_resized_crop_img.squeeze().numpy())
        ...     _ = cv2.waitKey(0)
        """
        img_shape = self.Shape(*img_shape)
        transforms_compose = list()
        transforms_compose.append(T.ToTensor())
        for method in self.methods:
            if method == 'random_resized_crop':
                transforms_compose.append(self._get_random_resized_crop(self.CROP_AREA, img_shape))
            elif method == 'random_affine':
                transforms_compose.append(self._get_random_affine(self.SHEAR_DEGREES))
            elif method == 'random_noise':
                transforms_compose.append(self._get_random_noise(self.WHITE_POINT_AREA))
        return T.Compose(transforms_compose)

    @staticmethod
    def _get_random_resized_crop(crop_area: float, img_shape: namedtuple) -> torch.nn.Module:
        h, w = img_shape.H, img_shape.W
        r = w / h
        transform = T.RandomResizedCrop(
            size=(h, w),
            scale=(crop_area, 1),
            ratio=(r, r),
            interpolation=T.InterpolationMode.NEAREST,
        )
        return transform

    @staticmethod
    def _get_random_affine(shear_degrees) -> torch.nn.Module:
        transform = T.RandomAffine(
            (0, 0), # no rotation rigid transformation
            shear=(0, shear_degrees)   # shearing on x axis
        )
        return transform

    @staticmethod
    def _get_random_noise(white_point_area: float) -> torch.nn.Module:
        transform = RandomNoise(
            p=0.5,
            scale=(white_point_area, white_point_area),
            ratio=(1, 1),
            erasing_area=150
        )
        return transform

    @staticmethod
    def _get_mix_up():
        pass

    @staticmethod
    def _get_cut_mix():
        pass


if __name__ == '__main__':
    import doctest
    doctest.testmod(verbose=True)