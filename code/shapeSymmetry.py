from skimage.io import *
from skimage.exposure import *
from skimage.filters import *
from skimage.measure import *
from skimage.transform import *
from skimage.segmentation import *

import matplotlib.pyplot as plt
import numpy as np

def pig2load(imNumber) :
    """Load a pigmented dermoscopic image from the PH2 Dataset.

    # Arguments :
        imNumber: String. The number of the image to be loaded (always with 3 digits : "IMD003").

    # Outputs :
        im: The loaded image.
    """

    filename = "../PH2Dataset/PH2 Dataset images/" + imNumber + "/" + imNumber + "_Dermoscopic_Image/" + imNumber + ".bmp"
    im = imread(filename)
    return im

def seg2load(imNumber) :
    """Load a segmented image from the PH2 Dataset.

    # Arguments :
        imNumber: String. The number of the image to be loaded (always with 3 digits : "IMD003").

    # Outputs :
        im: The loaded image.
    """

    filename = "../PH2Dataset/PH2 Dataset images/" + imNumber + "/" + imNumber + "_lesion/" + imNumber + "_lesion.bmp"
    im = imread(filename)
    return im

def symmetryRatios(segImage, stepAngle):
    """Calculate shape symmetry ratios over a range of angles from 0 to 180 degrees.

    # Arguments :
        segImage:  The segmented image whose shape symmetry is tested.
        stepAngle: Int. The step used to go from 0 to 180 degrees. Each angle permits to score symmetry in the
                       corresponding orientation.

    # Outputs :
        ratios: The list of symmetry ratios (scores) obtained over all angles tested.

    # Note on metric used to calculate scores :
        The Jaccard Index is used to perform symmetry calculus.
    """

    properties = regionprops(segImage)
    centroid = properties[0].centroid

    angles = [-k for k in range(0, 181, stepAngle)]
    ratios = [0] * len(angles)

    for angle in angles :

        rotIm = rotate(segImage, angle, resize=True, center=centroid)
        thresh = threshold_otsu(rotIm)
        rotIm = 1*(rotIm > thresh)

        properties = regionprops(rotIm)
        centroid = properties[0].centroid

        im2flip = rotIm[0:int(centroid[0]),0:np.shape(rotIm)[1]]
        flipIm = np.flip(im2flip, 0)

        lenIm2compare = np.shape(rotIm)[0] - int(centroid[0])


        if (lenIm2compare > np.shape(flipIm)[0]):
            black = np.zeros([np.shape(rotIm)[0] - int(centroid[0]), np.shape(rotIm)[1]])
            black[0:np.shape(flipIm)[0], 0:np.shape(rotIm)[1]] = flipIm
            flipIm = black
            im2compare = rotIm[int(centroid[0]):np.shape(rotIm)[0],0:np.shape(rotIm)[1]]

        else:
            black = np.zeros([int(centroid[0]),np.shape(rotIm)[1]])
            black[0:lenIm2compare , 0:np.shape(rotIm)[1]] = rotIm[int(centroid[0]):np.shape(rotIm)[0],0:np.shape(rotIm)[1]]
            im2compare = black

        histoComp = histogram(im2compare)
        histoFlip = histogram(flipIm)

        if histoComp[0][-1] > histoFlip[0][-1] :
            wPix = histoComp[0][-1]
        else :
            wPix = histoFlip[0][-1]

        join = join_segmentations(flipIm, im2compare)
        histoJoin = histogram(join)
        truePix = histoJoin[0][-1]

        ratio = truePix/wPix
        ratios[int(angle/stepAngle)] = 100*ratio

    return ratios

def symmetryShapeEval(segImage, stepAngle):
    """Evaluate the shape symmetry of an image over a range of angles from 0 to 180 degrees. There are 3 possibilities :
       symmetric (at least 2 axis), not fully symmetric (1-axis symmetry), or asymmetric.

    # Arguments :
        segImage:  The image whose shape symmetry is evaluated.
        stepAngle: Int. The step used to go from 0 to 180 degrees. Each angle permits to score symmetry in the
                       corresponding orientation

    # Outputs :
        res: list containing symmetry result (res[0]), percentage of symmetry of the main axe and angle from the
        horizontal (res[1]) if it exists, percentage of symmetry of the second main axe and angle from the
        horizontal (res[2]) if it exists.
                Symmetry result can be :
                        0 -> image shape symmetric
                        1 -> image shape not fully symmetric
                        2 -> image shape asymmetric
                        -1 -> unable to perform symmetry evaluation, image shape considered asymmetric
    """

    ratios = symmetryRatios(segImage, stepAngle)

    right = segImage[:, 0]
    left = segImage[:, np.shape(segImage)[1] - 1]
    up = segImage[0, :]
    bottom = segImage[np.shape(segImage)[0] - 1, :]

    mainCoef = max(ratios)
    highThresh = 92
    lowThresh = 90

    if (list(up).count(255) > np.shape(segImage)[1]/3 or list(bottom).count(255) > np.shape(segImage)[1]/3 or list(left).count(255) > np.shape(segImage)[0]/3 or list(right).count(255) > np.shape(segImage)[0]/3) :

        res = [-1, [None,None], [None,None]]

    elif (mainCoef > highThresh) :

        indMax = ratios.index(max(ratios))
        angleMax = stepAngle*indMax
        angleOrtho = angleMax + 90

        if angleOrtho<=180 and angleOrtho>= 0 :

            if ratios[int(angleOrtho/stepAngle)] < 88 :

                res = [1, [angleMax,mainCoef], [None,None]]

            else :

                res = [0, [angleMax,mainCoef], [angleOrtho,ratios[indMax+int(90/stepAngle)]]]

        else :
            angleOrtho -= 180
            if ratios[int(angleOrtho / stepAngle)] < 88:

                res = [1, [angleMax,mainCoef], [None,None]]

            else:
                if indMax+int(90/stepAngle)<len(ratios):
                    res = [0, [angleMax,mainCoef], [angleOrtho,ratios[indMax+int(90/stepAngle)]]]

                else :
                    res = [0, [angleMax, mainCoef], [angleOrtho, ratios[indMax - int(90 / stepAngle)]]]

    elif (mainCoef < lowThresh) :

        res = [2, [None,None], [None,None]]

    else:

        indMax = ratios.index(max(ratios))
        angleMax = stepAngle * indMax

        res = [1, [angleMax, mainCoef], [None, None]]

    return res

def displayShapesSymmetry(im, segIm, symmetry):
    """Display the axis of symmetry of an image, considering shape symmetry.

    # Arguments :
        im:       The image whose shape symmetry has been evaluated with the symmetryShapeEval() function.
        segIm:    The corresponding segmented image.
        symmetry: The output of the symmetryShapeEval() function used on `im`.

    # Outputs :
        Only display axis. Return 0 if no error occured.
    """

    fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
    fig.suptitle("Shape Symmetry", fontsize=20)

    properties = regionprops(segIm)
    centroid = properties[0].centroid



    if symmetry[0] == 0:
        pente = np.tan(symmetry[1][0] * np.pi / 180)
        ordOrig = centroid[0] - pente * centroid[1]
        x = np.linspace(0, np.shape(segIm)[1])
        y = pente * x + ordOrig

        penteOrtho = np.tan(symmetry[2][0] * np.pi / 180)
        ordOrigOrtho = centroid[0] - penteOrtho * centroid[1]
        xOrtho = np.linspace(0, np.shape(segIm)[1])
        yOrtho = penteOrtho * x + ordOrigOrtho

        axs[0].axis('off')
        axs[0].imshow(im, cmap=plt.cm.gray)
        axs[0].set_title('Input image')

        axs[1].axis('off')
        axs[1].imshow(segIm, cmap=plt.cm.gray)
        axs[1].set_title('Segmented image')

        axs[2].plot(x, y, "-r", linewidth=2)
        axs[2].plot(xOrtho, yOrtho, "-r", linewidth=0.8)
        axs[2].imshow(segIm, cmap=plt.cm.gray)
        axs[2].set_title("Main and second symmetry axes")
        axs[2].axis("off")
        plt.show()

    elif symmetry[0] == 1:
        pente = np.tan(symmetry[1][0] * np.pi / 180)
        ordOrig = centroid[0] - pente * centroid[1]
        x = np.linspace(0, np.shape(segIm)[1])
        y = pente * x + ordOrig

        axs[0].axis('off')
        axs[0].imshow(im, cmap=plt.cm.gray)
        axs[0].set_title('Input image')

        axs[1].axis('off')
        axs[1].imshow(segIm, cmap=plt.cm.gray)
        axs[1].set_title('Segmented image')

        axs[2].plot(x, y, "-r")
        axs[2].imshow(segIm, cmap=plt.cm.gray)
        axs[2].set_title("Main symmetry axis")
        axs[2].axis("off")
        plt.show()

    elif symmetry[0] == 2:

        axs[0].axis('off')
        axs[0].imshow(im, cmap=plt.cm.gray)
        axs[0].set_title('Input image')

        axs[1].axis('off')
        axs[1].imshow(segIm, cmap=plt.cm.gray)
        axs[1].set_title('Segmented image')

        axs[2].imshow(segIm, cmap=plt.cm.gray)
        axs[2].set_title("No symmetry axis")
        axs[2].axis("off")
        plt.show()

    else :

        axs[0].axis('off')
        axs[0].imshow(im, cmap=plt.cm.gray)
        axs[0].set_title('Input image')

        axs[1].axis('off')
        axs[1].imshow(segIm, cmap=plt.cm.gray)
        axs[1].set_title('Segmented image')

        axs[2].imshow(segIm, cmap=plt.cm.gray)
        axs[2].set_title("Too large lesion : no symmetry axis")
        axs[2].axis("off")
        plt.show()

    return 0

#----------EXAMPLE-------------------------
# segIm = seg2load("IMD035")
# im = pig2load("IMD035")
# sym = symmetryShapeEval(segIm, 3)
# print(sym)
# displayShapesSymmetry(im, segIm,sym)
#------------------------------------------