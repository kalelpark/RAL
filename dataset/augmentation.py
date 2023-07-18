from torchvision.transforms import transforms

def get_aug(args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_aug = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAdjustSharpness(sharpness_factor = 2),
        transforms.RandomErasing(),
        normalize,
        ])

    val_aug = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((args.img_size, args.img_size)),
        normalize,
        ])

    return train_aug, val_aug