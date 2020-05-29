from imdex.Captioner import Captioner

if __name__ == "__main__":
    cap = Captioner()

    images = [cap.load_image("./sample_img/cat.jpg"), cap.load_image("./sample_img/dog.jpg")]

    result = cap.captionize(images)
    
    print('Prediction Caption:')
    print(result)