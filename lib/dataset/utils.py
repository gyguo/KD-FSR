import torchvision.transforms as transforms


def gen_base_transform(cfg):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(cfg.DATA.INPUT_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomChoice([
            transforms.ColorJitter(brightness=0.5),
            transforms.ColorJitter(contrast=0.5),
            transforms.ColorJitter(saturation=0.5),
            transforms.ColorJitter(hue=0.5)]),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    test_transform = transforms.Compose([
        transforms.Resize((cfg.DATA.INPUT_SIZE, cfg.DATA.INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    return train_transform, test_transform


def gen_fsr_transform(cfg):
    train_transform_base = transforms.Compose([
        transforms.RandomResizedCrop(cfg.DATA.LARGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomChoice([
            transforms.ColorJitter(brightness=0.5),
            transforms.ColorJitter(contrast=0.5),
            transforms.ColorJitter(saturation=0.5),
            transforms.ColorJitter(hue=0.5)]),
    ])
    train_transform_l = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    train_transform_s = transforms.Compose([
        transforms.Resize(cfg.DATA.SMALL_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    test_transform_l = transforms.Compose([
        transforms.Resize((cfg.DATA.LARGE_SIZE, cfg.DATA.LARGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    test_transform_s = transforms.Compose([
        transforms.Resize((cfg.DATA.SMALL_SIZE, cfg.DATA.SMALL_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    return train_transform_base, train_transform_l, train_transform_s, test_transform_l, test_transform_s


def gen_fsr_i2_transform(cfg):
    train_transform_base = transforms.Compose([
        transforms.RandomResizedCrop(cfg.DATA.LARGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomChoice([
            transforms.ColorJitter(brightness=0.5),
            transforms.ColorJitter(contrast=0.5),
            transforms.ColorJitter(saturation=0.5),
            transforms.ColorJitter(hue=0.5)]),
    ])
    train_transform_l = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    train_transform_m = transforms.Compose([
        transforms.Resize(cfg.DATA.MIDDLE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    train_transform_s = transforms.Compose([
        transforms.Resize(cfg.DATA.SMALL_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    test_transform_l = transforms.Compose([
        transforms.Resize((cfg.DATA.LARGE_SIZE, cfg.DATA.LARGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    test_transform_m = transforms.Compose([
        transforms.Resize((cfg.DATA.MIDDLE_SIZE, cfg.DATA.MIDDLE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    test_transform_s = transforms.Compose([
        transforms.Resize((cfg.DATA.SMALL_SIZE, cfg.DATA.SMALL_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    return train_transform_base, train_transform_l, train_transform_m, train_transform_s, \
           test_transform_l, test_transform_m, test_transform_s