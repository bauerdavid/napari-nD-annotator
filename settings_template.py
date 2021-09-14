from skimage.data import cells3d
import tifffile
# path to *tif(f) file
image_path = None
test_image = tifffile.imread(image_path) if image_path else cells3d()
colormap = "magma"
# True if the data contains RGB channels along the last dimension
rgb = False
# True if any of the dimensions corresponds to channels
has_channels = True
# The dimension of the channels (should be None if rgb is True)
channels_dim = None if rgb or not has_channels\
    else test_image.shape.index(list(filter(lambda x: x<=3, test_image.shape))[0])
