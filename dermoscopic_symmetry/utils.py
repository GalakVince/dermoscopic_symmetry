import os

import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skimage import img_as_ubyte, img_as_float64
from skimage.draw import circle
from skimage.io import imread
from skimage.measure import regionprops, find_contours


def display_symmetry_axes(img, segmentation, symmetry_info):
    """Display the axis of symmetry of an image, considering shape symmetry.

    # Arguments :
        img:       The image whose shape symmetry has been evaluated with the symmetryShapeEval() function.
        segmentation:    The corresponding segmented image.
        symmetry_info: The output of the symmetryShapeEval() function used on `im`.
    """

    fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
    fig.suptitle("Shape Symmetry", fontsize=20)

    properties = regionprops(segmentation)
    centroid = properties[0].centroid



    if symmetry_info[0] == 0:
        pente = np.tan(symmetry_info[1][0] * np.pi / 180)
        ordOrig = centroid[0] - pente * centroid[1]
        x = np.linspace(0, np.shape(segmentation)[1])
        y = pente * x + ordOrig

        penteOrtho = np.tan(symmetry_info[2][0] * np.pi / 180)
        ordOrigOrtho = centroid[0] - penteOrtho * centroid[1]
        xOrtho = np.linspace(0, np.shape(segmentation)[1])
        yOrtho = penteOrtho * x + ordOrigOrtho

        axs[0].axis('off')
        axs[0].imshow(img, cmap=plt.cm.gray)
        axs[0].set_title('Input image')

        axs[1].axis('off')
        axs[1].imshow(segmentation, cmap=plt.cm.gray)
        axs[1].set_title('Segmented image')

        axs[2].plot(x, y, "-r", linewidth=2)
        axs[2].plot(xOrtho, yOrtho, "-r", linewidth=0.8)
        axs[2].imshow(segmentation, cmap=plt.cm.gray)
        axs[2].set_title("Main and secondary symmetry axes")
        axs[2].axis("off")
        plt.show()

    elif symmetry_info[0] == 1:
        pente = np.tan(symmetry_info[1][0] * np.pi / 180)
        ordOrig = centroid[0] - pente * centroid[1]
        x = np.linspace(0, np.shape(segmentation)[1])
        y = pente * x + ordOrig

        axs[0].axis('off')
        axs[0].imshow(img, cmap=plt.cm.gray)
        axs[0].set_title('Input image')

        axs[1].axis('off')
        axs[1].imshow(segmentation, cmap=plt.cm.gray)
        axs[1].set_title('Segmented image')

        axs[2].plot(x, y, "-r")
        axs[2].imshow(segmentation, cmap=plt.cm.gray)
        axs[2].set_title("Main symmetry axis")
        axs[2].axis("off")
        plt.show()

    elif symmetry_info[0] == 2:

        axs[0].axis('off')
        axs[0].imshow(img, cmap=plt.cm.gray)
        axs[0].set_title('Input image')

        axs[1].axis('off')
        axs[1].imshow(segmentation, cmap=plt.cm.gray)
        axs[1].set_title('Segmented image')

        axs[2].imshow(segmentation, cmap=plt.cm.gray)
        axs[2].set_title("No symmetry axis")
        axs[2].axis("off")
        plt.show()

    else:
        axs[0].axis('off')
        axs[0].imshow(img, cmap=plt.cm.gray)
        axs[0].set_title('Input image')

        axs[1].axis('off')
        axs[1].imshow(segmentation, cmap=plt.cm.gray)
        axs[1].set_title('Segmented image')

        axs[2].imshow(segmentation, cmap=plt.cm.gray)
        axs[2].set_title("Lesion too large: no symmetry axis")
        axs[2].axis("off")
        plt.show()


def load_dermoscopic(imNumber) :
    """Load a dermoscopic image from the PH2 Dataset.

    # Arguments :
        imNumber: String. The number of the image to be loaded (always with 3 digits : "IMD003").

    # Outputs :
        im: The loaded image.
    """
    filename = f"{package_path()}/data/PH2Dataset/PH2 Dataset images/{imNumber}/{imNumber}_Dermoscopic_Image/{imNumber}.bmp"
    try:
        im = imread(filename)
    except FileNotFoundError as exc:
        raise RuntimeError('Plase copy the PH2 dataset in /data/PH2Dataset/') from exc
    return im


def load_segmentation(imNumber):
    """Load a segmented image from the PH2 Dataset.

    # Arguments :
        imNumber: String. The number of the image to be loaded (always with 3 digits : "IMD003").

    # Outputs :
        im: The loaded image.
    """
    filename = f"{package_path()}/data/PH2Dataset/PH2 Dataset images/{imNumber}/{imNumber}_lesion/{imNumber}_lesion.bmp"
    try:
        im = imread(filename)
    except FileNotFoundError as exc:
        raise RuntimeError('Plase copy the PH2 dataset in /data/PH2Dataset/') from exc
    return im


def save_model(model, name):
    """Save a (Random Forest) classifier into a pickled file.

    # Arguments :
        model: Random Forest classifier.
        name: String. Filename, without extension.

    # Outputs :
        im: The loaded image.
    """
    dir = f'{package_path()}/data/models/'
    os.makedirs(dir, exist_ok=True)
    joblib.dump(model, f"{dir}/{name}.pkl")


def load_model(name):
    dir = f'{package_path()}/data/models/'
    return joblib.load(f"{dir}/{name}.pkl")


def load_PH2_asymmetry_GT():
    df = pd.read_excel(f"{package_path()}/data/symtab.xlsx")
    return df["Image Name"], df["Asymmetry"]


def package_path():
    return os.path.dirname(os.path.abspath(__file__))


def displayTextureSymmetry(im, segIm, symmetry):
    """Display the axis of symmetry of an image, considering textures symmetry.

    # Arguments :
        im:       The image whose textures symmetry has been evaluated.
        segIm:    The corresponding segmented image.
        symmetry: The output of the `symmetryTextureEval()` function.

    # Outputs :
        Display axis. Returns 0 if no error occured.
    """

    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    fig.suptitle("Texture Symmetry", fontsize=20)

    blkSeg = np.zeros((np.shape(segIm)[0] + 2, np.shape(segIm)[1] + 2))
    blkSeg[1:np.shape(blkSeg)[0] - 1, 1:np.shape(blkSeg)[1] - 1] = segIm
    segIm = blkSeg
    contour = find_contours(segIm, 0)
    cnt = contour[0]
    minx = min(cnt[:, 1])
    maxx = max(cnt[:, 1])
    miny = min(cnt[:, 0])
    maxy = max(cnt[:, 0])
    segIm = segIm[max(0, int(miny) - 1):int(maxy) + 1, max(0, int(minx) - 1):int(maxx) + 1]
    im = im[max(0, int(miny) - 1):int(maxy), max(0, int(minx) - 1):int(maxx) + 1]

    # Compute center of mass
    segIm = img_as_ubyte(segIm / 255)
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

        axs[1].plot(x, y, "-r", linewidth=2)
        axs[1].plot(xOrtho, yOrtho, "-r", linewidth=0.8)
        axs[1].imshow(im, cmap=plt.cm.gray)
        axs[1].set_title("Main symmetry axis")
        axs[1].axis("off")
        plt.show()

    elif symmetry[0] == 1:
        pente = np.tan(symmetry[1][0] * np.pi / 180)
        ordOrig = centroid[0] - pente * centroid[1]
        x = np.linspace(0, np.shape(segIm)[1])
        y = pente * x + ordOrig

        axs[0].axis('off')
        axs[0].imshow(im, cmap=plt.cm.gray)
        axs[0].set_title('Input image')

        axs[1].plot(x, y, "-r")
        axs[1].imshow(im, cmap=plt.cm.gray)
        axs[1].set_title("Main symmetry axis")
        axs[1].axis("off")
        plt.show()

    else:

        axs[0].axis('off')
        axs[0].imshow(im, cmap=plt.cm.gray)
        axs[0].set_title('Input image')

        axs[1].imshow(im, cmap=plt.cm.gray)
        axs[1].set_title("No symmetry axis")
        axs[1].axis("off")
        plt.show()


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