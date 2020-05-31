from imdex.Indexer import Indexer

if __name__ == "__main__":
    
    idr = Indexer(captions_csv_path="captions.csv")

    images = [idr.load_image("./sample_img/surf.jpg")]

    idr.add_images(images, ['surf'])

    print(idr.query("a cat is laying"))