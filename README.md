# dermoscopic_symmetry

dermoscopic_symmetry is a Python package aiming at study skin lesion's symmetry (regarding shape and textures)
and help the diagnose of diseases like melanomas. <br/>Basically, the symmetry study is divided into
two parts: shapes symmetry and textures symmetry. Here, shapes means the aspect of the outskirts of a lesion
and its global form whereas textures stand for colors and types of perceived textures.

**Note :** The package has been built referring to the [PH² Dataset](https://www.fc.up.pt/addi/ph2%20database.html).
See :

[Teresa Mendonça, Pedro M. Ferreira, Jorge Marques, Andre R. S. Marcal, 
Jorge Rozeira. PH² - A dermoscopic image database for research and 
benchmarking, 35th International Conference of the IEEE Engineering in 
Medicine and Biology Society, July 3-7, 2013, Osaka, Japan.](https://ieeexplore.ieee.org/document/6610779?tp=&arnumber=6610779&url=http:%2F%2Fieeexplore.ieee.org%2Fxpls%2Fabs_all.jsp%3Farnumber%3D6610779)

To be able to use it directly and properly, you must download and have 
access to it.
 
 ## Installation
 
 Use [pip](https://pip.pypa.io/en/stable/) to install the package.
 
 ```bash
 pip install dermoscopic_symmetry
 ```
 These are the Python files used to study the symmetry of skin lesions from the PH² Dataset
  (see Usage section):

1. **`shape_symmetry.py`** : containing functions to study the symmetry of shapes in a lesion's image.

2. **`classifier_feeder.py`** : containing functions to create a classifier able to recognize if 2 patches taken in a lesion's
 image are similar or not.<br/> A new dataset called `patchesDataSet`, derivating from the PH² Dataset, has been designed to 
 train this classifier. It is composed of patches pairs taken in the PH² Dataset images with 
 one half similar and the other non similar.
 
3. **`patches_for_texture_symmetry.py`** : containing functions to take patches from a dermoscopic image and extract
features from them (textures and color).

4. **`texture_symmetry.py`** : containing functions using the previous classifier and features
to study the symmetry of textures in a lesion's image.

5. **`combined_classifier.py`** : containing functions using only shape features, only textures
features or both of them to train classifiers and be able to know which one is the best
according to expert diagnose in the PH² Dataset.
<br/> Those classifiers are trained according to the `ShapesScores.csv`, `TextureScores.csv` and
`ShapeAndTextureScores.csv` files contained in the `data` repository. The final models are saved as `shapeModel.pkl`, `textureModel.pkl` and 
`shapeAndTextureModel.pkl` in the `data/models` repository.

**Note :** The code used to create the 
patchesDataSet is given in the **`patches_dataset_creator.py`** file. The **`utils.py`** file contains the utilities 
functions.

 ## Usage

Each code script has an :
`example()` function at the beginning aiming at presenting its functionalities. This function is run
as a default `main`.

## License

[MIT](https://choosealicense.com/licenses/mit/)
