from imdex.Captioner import Captioner
from imdex.Indexer import Indexer

if __name__ == "__main__":
    
    idr = Indexer()

    cap = Captioner()

    images = [cap.load_image("./sample_img/cat.jpg"), cap.load_image("./sample_img/dog.jpg")]

    idr.add_images(images, ['cat', 'dog'])

    print(idr.query("a cat is laying"))