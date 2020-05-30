from imdex.Captioner import Captioner
from imdex.Indexer import Indexer

if __name__ == "__main__":
    
    idr = Indexer()
    
    cap = Captioner()

    images = [cap.load_image("./sample_img/cat.jpg"), cap.load_image("./sample_img/dog.jpg")]

    #result = cap.captionize(images)

    res = idr.add_images(images, "")
    print(res.shape)
    
    #print('Prediction Caption:')
    #print(result)