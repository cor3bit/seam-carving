# Seam carving experiments

Replication of the algorithms described 
in the paper
[Seam Carving for Content-Aware Image Resizing](https://perso.crans.org/frenoy/matlab2012/seamcarving.pdf) 
by Avidan and Shamir.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Usage

To run the algorithm:

```run
python seam_carving.py
```

## Results

Results for seam removal:

Input Image            |  Output Image     |
:-------------------------:|:-------------------------:|
![](input/fig5/waterfall.png)  |  ![](output/fig5.png) |

Results for seam insertion:

Input Image             |  Output Image    |
:-------------------------:|:-------------------------:|
![](input/fig8/dolphin.jpg)  |  ![](output/fig8e.png) |