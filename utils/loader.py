import os

from dataset import DataLoaderTrain, DataLoaderVal, DataLoaderTestAhsan
def get_training_data(rgb_dir, img_options):
    print(rgb_dir)
    assert os.path.exists(rgb_dir)
    return DataLoaderTrain(rgb_dir, img_options, None)

def get_validation_data(rgb_dir):
    print(rgb_dir)
    assert os.path.exists(rgb_dir)
    img_options_train = {'patch_size':256}

    # return DataLoaderTestAhsan(rgb_dir, img_options_train, None)
    return DataLoaderVal(rgb_dir,img_options_train, None)
