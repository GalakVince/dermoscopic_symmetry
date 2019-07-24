import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import img_as_float64, img_as_ubyte
from skimage.draw import circle
from skimage.filters import threshold_otsu
from skimage.measure import regionprops, find_contours
from skimage.transform import rotate

from dermoscopic_symmetry.classifier_feeder import classifierTrainer, dataExtractorForTraining
from dermoscopic_symmetry.patches_for_symmetry import textureDataExtractor
from dermoscopic_symmetry.utils import load_dermoscopic, load_segmentation, displayTextureSymmetry


def example(create_features=True):
    """Usage example of the main functionalities within this file. """
    im = load_dermoscopic("IMD009")
    segIm = load_segmentation("IMD009")

    if create_features:
        dataExtractorForTraining(patchesPerImage=10, nbImages=199, nbBins=4)
        clf, acc = classifierTrainer(100)

    else:
        raise NotImplementedError

    res, ratios = symmetryTextureEval(im, segIm, 5)
    print(ratios)
    displayTextureSymmetry(im, segIm, res)

    preds, nonSimilar, similar = symmetryTexturePred(clf)
    patches, points, reference = textureDataExtractor(im, segIm, 32, 4)
    displaySimilarityMatches(im, segIm, preds, points, reference)


def symmetryTexturePred(classifier):
    """Predict if symetric pairs of patches taken in a dermoscopic image are similar or not using features extracted
       with the `textureDataExtractor()` function and stored in the "feature.csv" file.

    # Arguments :
        classifier:  The trained random forest classifier (with patchesDataSet).

    # Outputs :
        preds:         The predictions (0 if non similar, 1 if similar).
        nonSimilarNum: Int. The number of non similar matches.
        similarNum:    Int. The number of similar matches.
    """

    data = pd.read_csv(f"{package_path()}/patchesDataSet/features.csv")
    features = list(data)
    del features[0]

    toPredict = data[features]
    preds = classifier.predict(toPredict)

    nonSimilarNum = list(preds).count(0)
    similarNum = list(preds).count(1)

    return preds, nonSimilarNum, similarNum


def symmetryTextureEval(im, segIm, stepAngle):
    """Evaluate the textures symmetry of an image over a range of angles from 0 to 180 degrees. There are 3
       possibilities : symmetric (at least 2 axis), not fully symmetric (1-axis symmetry), or asymmetric.

    # Arguments :
        im:        The dermoscopic image whose textures symmetry evaluated.
        segIm:     The corresponding segmented image.
        stepAngle: Int. The step used to go from 0 to 180 degrees. Each angle permits to score symmetry in the
                   corresponding orientation

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

    classifier, accScore = classifierTrainer(100)

    angles = [-k for k in range(0, 181, stepAngle)]
    simRatios = []
    for angle in angles :
        rotSegIm = img_as_ubyte(rotate(segIm, angle, resize=True, center=originalCentroid))
        thresh = threshold_otsu(rotSegIm)
        rotSegIm = 255*(rotSegIm > thresh)

        properties = regionprops(rotSegIm)
        centroid = properties[0].centroid
        rotIm = rotate(im, angle, resize=True, center=centroid)

        textureDataExtractor(rotIm, rotSegIm, 32, 4)
        preds, nonSimilarNum, similarNum = symmetryTexturePred(classifier)
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


def displaySimilarityMatches(im, segIm, preds, points, reference):
    """Display the map of similar and non similar matches over the original image thanks to respectively green and red
       circles.

    # Arguments :
        im:        The image whose textures symmetry has been evaluated.
        segIm:     The corresponding segmented image.
        preds:     The predictions given by the `symmetryTexturePred()` function.
        points:    The list of points correspondind to used patches in the image (`textureDataExtractor()` function).
        reference: The part of the image taken as a reference ("Upper" or "Lower") (`textureDataExtractor()` function).

    # Outputs :
        Display the map of similarity.
    """

    # Crop images to be centered on the lesion
    blkSeg = np.zeros((np.shape(segIm)[0]+2,np.shape(segIm)[1]+2))
    blkSeg[1:np.shape(blkSeg)[0]-1,1:np.shape(blkSeg)[1]-1] = segIm
    segIm = blkSeg
    contour = find_contours(segIm, 0)
    cnt = contour[0]
    minx = min(cnt[:, 1])
    maxx = max(cnt[:, 1])
    miny = min(cnt[:, 0])
    maxy = max(cnt[:, 0])
    segIm = segIm[max(0, int(miny)-1):int(maxy)+1, max(0, int(minx)-1):int(maxx)+1]
    im = im[max(0, int(miny)-1):int(maxy), max(0, int(minx)-1):int(maxx)+1]

    # Compute center of mass
    segIm = img_as_ubyte(segIm/255)
    properties = regionprops(segIm)
    centroid = properties[0].centroid



    # Define all necessary variables
    blend = im.copy()
    alpha = 0.6
    blend = img_as_float64(blend)
    patchSize = 32
    index = 0

    # Calculate the real coordinates (in the original image) of points from the lower part
    lowPoints = []
    for pt in points:
        lowPoints.append([np.shape(im)[0] - pt[0], pt[1]])


    if reference == "Upper":

        # If the reference is the upper part, symetric points has to be calculated from points
        for point in points:
            rrUp, ccUp = circle(point[0] - 0.5 + int(patchSize / 2), point[1] - 0.5 + int(patchSize / 2), int(patchSize / 2))
            symPoint = [point[0] + 2*(abs(centroid[0]-point[0])), point[1]]
            rrLow, ccLow = circle(symPoint[0] + 0.5 - int(patchSize/2), symPoint[1] - 0.5 + int(patchSize/2),int(patchSize/2))

            # Apply green or red circles according to predictions
            if preds[index] == 1 :
                greenFilter = im * 0
                greenFilter = img_as_ubyte(greenFilter)
                # Rectangular shape
                # greenFilter[point[0]:point[0]+patchSize,point[1]:point[1]+patchSize,1] = 255
                # Circular shape
                greenFilter[rrLow, ccLow, 1] = 200
                greenFilter[rrUp, ccUp, 1] = 200

                greenFilter = img_as_float64(greenFilter)
                mask = greenFilter != 0

                blend[mask] = blend[mask]*0.9 + greenFilter[mask] * (1-alpha)

            else :
                redFilter = im * 0
                redFilter = img_as_ubyte(redFilter)
                # Rectangular shape
                # redFilter[point[0]:point[0]+patchSize,point[1]:point[1]+patchSize,0] = 255
                # Circular shape
                redFilter[rrLow, ccLow, 0] = 200
                redFilter[rrUp, ccUp, 0] = 200

                redFilter = img_as_float64(redFilter)

                mask = redFilter != 0

                blend[mask] = blend[mask] * 0.9 + redFilter[mask] * (1 - alpha)

            index += 1

    else:

        # If the reference is the lower part, symetric points has to be calculated from lowPoints
        for point in lowPoints:
            rrLow,ccLow = circle(point[0] + 0.5 - int(patchSize/2), point[1] - 0.5 + int(patchSize/2),int(patchSize/2))
            symPoint = [point[0] - 2 * (abs(centroid[0] - point[0])), point[1]]
            rrUp, ccUp = circle(symPoint[0] - 0.5 + int(patchSize/2), symPoint[1] - 0.5 + int(patchSize/2),int(patchSize/2))

            # Apply green or red circles according to predictions
            if preds[index] == 1 :
                greenFilter = im * 0
                greenFilter=img_as_ubyte(greenFilter)
                # Rectangular shape
                # greenFilter[point[0]:point[0]+patchSize,point[1]:point[1]+patchSize,1] = 255
                # Circular shape
                greenFilter[rrLow, ccLow, 1] = 200
                greenFilter[rrUp, ccUp, 1] = 200

                greenFilter = img_as_float64(greenFilter)
                mask = greenFilter != 0

                blend[mask] = blend[mask]*0.9 + greenFilter[mask] * (1-alpha)

            else :
                redFilter = im * 0
                redFilter = img_as_ubyte(redFilter)
                # Rectangular shape
                # redFilter[point[0]:point[0]+patchSize,point[1]:point[1]+patchSize,0] = 255
                # Circular shape
                redFilter[rrLow, ccLow, 0] = 200
                redFilter[rrUp, ccUp, 0] = 200

                redFilter = img_as_float64(redFilter)
                mask = redFilter != 0

                blend[mask] = blend[mask] * 0.9 + redFilter[mask] * (1 - alpha)

            index += 1

    # Display original image and the map with the axis tested in blue
    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    fig.suptitle("Texture Symmetry", fontsize=20)

    axs[0].axis('off')
    axs[0].imshow(im, cmap=plt.cm.gray)
    axs[0].set_title('Input image for textures symmetry')

    axs[1].axis('off')
    x = np.linspace(0, np.shape(im)[1])
    y = 0 * x + centroid[0]
    plt.plot(x,y,"-b")
    plt.text(5,centroid[0]-5,"Tested axe", fontsize= 10,color="b")
    axs[1].imshow(blend, cmap=plt.cm.gray)
    axs[1].set_title('Similar matches (green) and non similar matches (red)')

    plt.show()


# Run example() whenever running this script as main
if __name__ == '__main__':
    example()
