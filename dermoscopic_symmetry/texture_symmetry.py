import numpy as np
import pandas as pd
from skimage import img_as_ubyte
from skimage.filters import threshold_otsu
from skimage.measure import regionprops
from skimage.transform import rotate

from dermoscopic_symmetry.classifier_feeder import classifierTrainer, dataExtractorForTraining
from dermoscopic_symmetry.patches_for_texture_symmetry import texture_symmetry_features
from dermoscopic_symmetry.utils import load_dermoscopic, load_segmentation, package_path, \
    display_similarity_matches, display_symmetry_axes, load_model


def example(retrain_model=False, sample_name='IMD400'):
    """Usage example of the main functionalities within this file. """
    img = load_dermoscopic(sample_name)
    segm = load_segmentation(sample_name)

    if retrain_model:
        dataExtractorForTraining(patchesPerImage=10, nbImages=199, nbBins=4)
        clf, acc = classifierTrainer(200)

    else:
        clf = load_model('PatchClassifierModel')

    symmetry_info, ratios = texture_symmetry(img, segm, stepAngle=20, classifier=clf)
    display_symmetry_axes(img, segm, symmetry_info, title='Texture symmetry')

    display_similarity_matches(img, segm, patchSize=32, nbBins=4, classifier=clf,
                               axis_in_degrees=symmetry_info[1][0])


def texture_symmetry_predict_patches(classifier, data=None, data_backup_file='FeaturesForPreds'):
    """Predict if symetric pairs of patches taken in a dermoscopic image are similar or not using features extracted
       with the `texture_symmetry_features()` function and stored in the "FeatureForPreds.csv" file.

    # Arguments :
        classifier:  The trained random forest classifier (with patchesDataSet).
        data:   As returned by the texture_symmetry_features function (optional).
        data_backup_filename:   Only if data is None, file to load data from.

    # Outputs :
        preds:         The predictions (0 if non similar, 1 if similar).
        nonSimilarNum: Int. The number of non similar matches.
        similarNum:    Int. The number of similar matches.
    """
    if data is None:
        data = pd.read_csv(f"{package_path()}/data/patchesDataSet/{data_backup_file}.csv", index_col=False)
        features = list(data)
        del features[0]
    else:
        features = list(data)

    toPredict = data[features]
    preds = classifier.predict(toPredict)

    nonSimilarNum = list(preds).count(0)
    similarNum = list(preds).count(1)

    return preds, nonSimilarNum, similarNum


def texture_symmetry(im, segIm, stepAngle, classifier=None):
    """Evaluate the textures symmetry of an image over a range of angles from 0 to 180 degrees. There are 3
       possibilities : symmetric (at least 2 axis), not fully symmetric (1-axis symmetry), or asymmetric.

    # Arguments :
        im:        The dermoscopic image whose textures symmetry evaluated.
        segIm:     The corresponding segmented image.
        stepAngle: Int. The step used to go from 0 to 180 degrees. Each angle permits to score symmetry in the
                   corresponding orientation
        classifier: Classifier (e.g. from load_model) or None to retrain

    # Outputs :
        res:      List containing symmetry result (res[0]), percentage of symmetry of the main axe and angle from the
                  horizontal (res[1]) if it exists, percentage of symmetry of the second main axe and angle from the
                  horizontal (res[2]) if it exists.
                     Symmetry result can be :
                            0 -> image textures symmetric
                            1 -> image textures not fully symmetric
                            2 -> image textures asymmetric
    simRatios: List containing textures symmetry score for each angle tested.

    # Note on metric used to calculate scores :
        The number of similar matches over the whole matches is used to perform symmetry calculus.
    """

    properties = regionprops(segIm)
    originalCentroid = properties[0].centroid

    if not classifier:
        classifier, _ = classifierTrainer(100)

    angles = [-k for k in range(0, 181, stepAngle)]
    simRatios = []
    for angle in angles :
        rotSegIm = img_as_ubyte(rotate(segIm, angle, resize=True, center=originalCentroid))
        thresh = threshold_otsu(rotSegIm)
        rotSegIm = 255*(rotSegIm > thresh)

        properties = regionprops(rotSegIm)
        centroid = properties[0].centroid
        rotIm = rotate(im, angle, resize=True, center=centroid)

        _, _, _, data = texture_symmetry_features(rotIm, rotSegIm, 32, 4)
        preds, nonSimilarNum, similarNum = texture_symmetry_predict_patches(classifier, data=data)
        simRatios.append(similarNum/(nonSimilarNum+similarNum))

    ind = simRatios.index(max(simRatios))

    if simRatios[ind] >= 0.77 and np.mean(simRatios) >= 0.55:

        if ind + int(90 / stepAngle) < len(angles):
            indOrtho = ind + int(90 / stepAngle)
        else:
            indOrtho = ind - int(90 / stepAngle)

        if simRatios[indOrtho] >= 0.70:
            res = [0, [ind*stepAngle, simRatios[ind]], [indOrtho*stepAngle, simRatios[indOrtho]]]
        else :
            res = [1, [ind * stepAngle, simRatios[ind]], [None,None]]

    elif simRatios[ind] <= 0.70:

        if min(simRatios) <= 0.45:
            res = [2, [None,None], [None,None]]
        else :
            res = [1, [ind * stepAngle, simRatios[ind]], [None, None]]

    else :

        res = [1, [ind * stepAngle, simRatios[ind]], [None,None]]

    return (res,simRatios)


# Run example() whenever running this script as main
if __name__ == '__main__':
    example()
