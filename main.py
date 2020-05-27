from imdex.Captioner import Captioner

if __name__ == "__main__":
    cap = Captioner()

    result = cap.captionize()
    
    print('Prediction Caption:')
    print(' '.join(result))
    