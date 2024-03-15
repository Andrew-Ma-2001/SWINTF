import os



def _get_all_images(path):
    """
    Get all images from a path.
    """
    # Write a warning if the path is not exist
    if not os.path.exists(path):
        raise ValueError('Path [{:s}] not exists.'.format(path))

    images = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith('.png') or file.endswith('.jpg'):
                images.append(os.path.join(root, file))

    # Sort the images
    images = sorted(images)
    return images

def get_all_images(path):
    paths = None
    if isinstance(path, str):
        paths = _get_all_images(path)
    elif isinstance(path, list):
        paths = []
        for p in path:
            paths += _get_all_images(p)
    else:
        raise ValueError('Wrong path type: [{:s}].'.format(type(path)))
    return paths