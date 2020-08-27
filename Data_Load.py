import torchvision
Train_data = torchvision.datasets.VOCDetection(root = './data',
                                               year = '2012',
                                               image_set = 'train',
                                               download = True,
                                               transform = None,
                                               target_transform = None,
                                               transforms = None)
