import albumentations as albu


def get_tr_augmentation(img_size):
    augmentations = [

        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        albu.PadIfNeeded(min_height=img_size[0], min_width=img_size[1], always_apply=True, border_mode=0),
        albu.RandomCrop(height=img_size[0], width=img_size[1], always_apply=True),  

        albu.IAAAdditiveGaussianNoise(p=0.2),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        )
    ]

    return augmentations


# When loading surface normals, 
# augmentations which apply affine transformations are not used
def get_tr_augmentation_normals(img_size):
    augmentations = [

        albu.PadIfNeeded(min_height=img_size[0], min_width=img_size[1], always_apply=True, border_mode=0),
        albu.RandomCrop(height=img_size[0], width=img_size[1], always_apply=True), 

        albu.IAAAdditiveGaussianNoise(p=0.2),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        )
    ]

    return augmentations
