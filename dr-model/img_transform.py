from albumentations.pytorch import ToTensorV2
import albumentations as A

# Image Augmentations

train_transforms = A.Compose([
    A.Resize(width=760, height=760),
    A.RandomCrop(height=728, width=728),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Blur(p=0.3),
    A.CLAHE(p=0.3),
    A.ColorJitter(p=0.3),
    A.CoarseDropout(max_holes=12, max_height=20, max_width=20, p=0.3),
    A.Affine(shear=30, rotate=0, p=0.2, mode=0),
    A.Normalize(
        mean=[0.3199, 0.2240, 0.1609],
        std=[0.3020, 0.2183, 0.1741],
        max_pixel_value=255.0,
    ),
    ToTensorV2(),
])

val_transforms = A.Compose([
    A.Resize(height=728, width=728),
    A.Normalize(
        mean=[0.3199, 0.2240, 0.1609],
        std=[0.3020, 0.2183, 0.1741],
        max_pixel_value=255.0,
    ),
    ToTensorV2(),
])