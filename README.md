# imdex
Imdex is a library that allows semantic searches over images sets

# How to use

## Installing

To install the library just run the following command on your 

```
pip install imdex
```

## Modules

Once the library is installed you just have to import the modules you that are going to use, the available modules are listed below.

```
Imdex
  |-Loader
  |-Captioner
  \-Indexer
```

### Loader

The Loader module is a simples way to import and format all images that are going to be indexed, it allows you to import single images or folders, it returns the images correctly resize and ready to be used

```
from imdex.loader import load_image, load_foder

images = [load_image("./sample_img/surf.jpg")]#importing single image
images, references = load_foder("sample_img")#importing all images in a folder with their names to be used as references to the image
```

### Captioner

This module is responsible for captioning the images, it is called by the indexer module, but can be imported and used independently

```
from imdex.Captioner import Captioner

cap = Captioner()

images = []#list of images

cap.captionize(images)# returns a list of captions
```

### Indexer

The Indexer is the class responsible for retrieving images, once the image is included in the indexer the Captioner module will provide a description in english that can be saved as a file and loaded later, all images need a string reference to identify them, as their names for example. To query the images their descriptions and the queried sentence are converted to a vector of embeddings to measure the distance between those, when the distances are calculated a sorted vector is created and returned as output of the function.

```
from imdex.Indexer import Indexer

idr = Indexer()

images, references = load_foder("sample_img")

idr.add_images(images, references)

idr.query("a cat is laying")

idr.save_to_csv("images.imdex")
```

To load the descriptions you have to create a new instance of the Indexer class and use the name of the file as an argument in the constructor

```
idr = Indexer(captions_csv_path="images.imdex")
```

# Future improvements

As a new library it has a lot a improvements to be done, and here is an itemized list of the main ones:

- Query time optimization
- Multi language queries support
- Improvements on the captioning model
- Enable the use of custom word embedding models on search

Wanna help? Send a merge request or an email to lucv.frot@gmail.com
