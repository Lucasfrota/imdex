from imdex.Indexer import Indexer
from imdex.loader import load_image, load_foder

if __name__ == "__main__":
    
    idr = Indexer()

    #images = [load_image("./sample_img/surf.jpg")]
    images, references = load_foder("sample_img")

    idr.add_images(images, references)

    print(idr.query("a cat is laying"))