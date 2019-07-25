import numpy as np
from skimage.exposure import histogram
from skimage.filters import threshold_otsu
from skimage.measure import regionprops
from skimage.segmentation import join_segmentations
from skimage.transform import rotate


def example():
    """Usage example of the main functionalities within this file. """
    from dermoscopic_symmetry.utils import load_dermoscopic, load_segmentation, display_symmetry_axes
    dermoscopic = load_dermoscopic("IMD400")
    segmentation = load_segmentation("IMD400")
    symmetry_info = shape_symmetry(segmentation, angle_step=9)
    print(f'Symmetry info: {symmetry_info}')
    display_symmetry_axes(dermoscopic, segmentation, symmetry_info,
                          title='Shape symmetry')


def shape_symmetry_ratios(segmentation, angle_step=9):
    """Calculate shape symmetry ratios over a range of angles from 0 to 180 degrees.

    # Arguments :
        segmentation:  The segmented image whose shape symmetry is tested.
        angle_step: Int. The step used to go from 0 to 180 degrees. Each angle permits to score symmetry in the
                       corresponding orientation.

    # Outputs :
        ratios: The list of symmetry ratios (scores) obtained over all angles tested.

    # Note on metric used to calculate scores :
        The Jaccard Index is used to perform symmetry calculus.
    """

    properties = regionprops(segmentation)
    centroid = properties[0].centroid

    angles = [-k for k in range(0, 181, angle_step)]
    ratios = [0] * len(angles)

    for angle in angles :

        rotIm = rotate(segmentation, angle, resize=True, center=centroid)
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
        ratios[int(angle/angle_step)] = 100*ratio

    return ratios


def shape_symmetry(segmentation, angle_step=9):
    """Evaluate the shape symmetry of an image over a range of angles from 0 to 180 degrees. There are 3 possibilities :
       symmetric (at least 2 axis), not fully symmetric (1-axis symmetry), or asymmetric.

    # Arguments :
        segmentation:  The image whose shape symmetry is evaluated.
        angle_step: Int. The step used to go from 0 to 180 degrees. Each angle permits to score symmetry in the
                       corresponding orientation

    # Outputs :
        symmetry_info: list containing symmetry result (symmetry_info[0]), percentage of symmetry of the main axe and angle from the
            horizontal (symmetry_info[1]) if it exists, percentage of symmetry of the second main axe and angle from the
            horizontal (symmetry_info[2]) if it exists.
                Symmetry result can be :
                         0 -> image shape symmetric
                         1 -> image shape not fully symmetric
                         2 -> image shape asymmetric
                        -1 -> unable to perform symmetry evaluation, image shape considered asymmetric
    """

    ratios = shape_symmetry_ratios(segmentation, angle_step)

    right = segmentation[:, 0]
    left = segmentation[:, np.shape(segmentation)[1] - 1]
    up = segmentation[0, :]
    bottom = segmentation[np.shape(segmentation)[0] - 1, :]

    mainCoef = max(ratios)
    highThresh = 92
    lowThresh = 90

    if (list(up).count(255) > np.shape(segmentation)[1]/3 or list(bottom).count(255) > np.shape(segmentation)[1]/3 or list(left).count(255) > np.shape(segmentation)[0]/3 or list(right).count(255) > np.shape(segmentation)[0]/3) :

        symmetry_info = [-1, [None,None], [None,None]]

    elif (mainCoef > highThresh) :

        indMax = ratios.index(max(ratios))
        angleMax = angle_step*indMax
        angleOrtho = angleMax + 90

        if angleOrtho<=180 and angleOrtho>= 0 :

            if ratios[int(angleOrtho/angle_step)] < 88 :

                symmetry_info = [1, [angleMax,mainCoef], [None,None]]

            else :

                symmetry_info = [0, [angleMax,mainCoef], [angleOrtho,ratios[indMax+int(90/angle_step)]]]

        else :
            angleOrtho -= 180
            if ratios[int(angleOrtho / angle_step)] < 88:

                symmetry_info = [1, [angleMax,mainCoef], [None,None]]

            else:
                if indMax+int(90/angle_step)<len(ratios):
                    symmetry_info = [0, [angleMax,mainCoef], [angleOrtho,ratios[indMax+int(90/angle_step)]]]

                else :
                    symmetry_info = [0, [angleMax, mainCoef], [angleOrtho, ratios[indMax - int(90 / angle_step)]]]

    elif (mainCoef < lowThresh) :

        symmetry_info = [2, [None,None], [None,None]]

    else:

        indMax = ratios.index(max(ratios))
        angleMax = angle_step * indMax

        symmetry_info = [1, [angleMax, mainCoef], [None, None]]

    return symmetry_info


# Run example() whenever running this script as main
if __name__ == '__main__':
    example()
