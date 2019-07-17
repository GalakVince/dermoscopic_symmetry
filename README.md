# skin_sym_pkg

skin_sym_pkg is a Python package aiming at study skin lesion's symmetry (regarding shape and textures)
and help the diagnose of diseases like melanomas. <br/>Basically, the symmetry study is divided into
two parts: shapes symmetry and textures symmetry. Here, shapes means the aspect of the outskirts of a lesion
and its global form whereas textures stand for colors and types of perceived textures.

**Note :** The package has been built referring to the [PH² Dataset](https://www.fc.up.pt/addi/ph2%20database.html).
See :

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Teresa Mendonça, Pedro M. Ferreira, Jorge Marques, Andre R. S. Marcal, 
Jorge Rozeira. PH² - A dermoscopic image &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;database for research and 
benchmarking, 35th International Conference of the IEEE Engineering in 
Medicine and &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Biology Society, July 3-7, 2013, Osaka, Japan.](https://ieeexplore.ieee.org/document/6610779?tp=&arnumber=6610779&url=http:%2F%2Fieeexplore.ieee.org%2Fxpls%2Fabs_all.jsp%3Farnumber%3D6610779)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;To be able to use it directly and properly, you must download and have 
access to it.
 
 ## Installation
 
 Use [pip](https://pip.pypa.io/en/stable/) to install the package.
 
 ```bash
 pip install skin_sym_pkg
 ```
 These are the Python files (in the `code` repository) used to study the symmetry of skin lesions from the PH² Dataset
  (see Usage section):

1. **`shapeSymmetry.py`** : containing functions to study the symmetry of shapes in a lesion's image.

2. **`classifierFeeder.py`** : containing functions to create a classifier able to recognize if 2 patches taken in a lesion's
 image are similar or not.<br/> A new dataset called `patchesDataSet`, derivating from the PH² Dataset, has been designed to 
 train this classifier. It is composed of patches pairs taken in the PH² Dataset images with 
 one half similar and the other non similar.
 
3. **`patchesForSymmetry.py`** : containing functions to take patches from a dermoscopic image and extract
features from them (textures and color).

4. **`textureSymmetry.py`** : containing functions using the previous classifier and features
to study the symmetry of textures in a lesion's image.

5. **`finalClassifier.py`** : containing functions using only shape features, only textures
features or both of them to train classifiers and be able to know which one is the best
according to expert diagnose in the PH² Dataset.
<br/> Those classifiers are trained according to the `ShapesScores.csv`, `TextureScores.csv` and
`ShapeAndTextureScores.csv` files. The final models are saved as `shapeModel.pkl`, `textureModel.pkl` and 
`shapeAndTextureModel.pkl`.

 ## Usage

Each code script has an :
```python
#--------EXAMPLE--------
# 
# 
#-----------------------
```

at the end aiming at showing how to use the designed functions. <br/>The code used to create the 
patchesDataSet and the `*Scores.csv` files are given in the `code/datasetCreators` repository. Note that the 
`ShapeAndTextureScores.csv` file had been handly created merging `ShapesScores.csv` and `TextureScores.csv`.

**Note :** To use the `datasetCreators` functions, a `symtab.xlsx` file must be construct and add 
to the same repository level as &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;the `*.pkl` files are (package's top level). This file is
an excel file containing two columns : the first contains all 
the &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"Image Name" column of the `PH2_dataset.xlsx` file (available once the PH² Dataset has
been downloaded) and the second one contains all the "Asymmetry"
column of the `PH2_dataset.xlsx` file.


## License

[MIT](https://choosealicense.com/licenses/mit/)