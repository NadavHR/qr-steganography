import cv2
import encoding
import qrcode


opacity = 0.01
encoded_window = "encoded"
decoded_window = "decoded"
def main():
    cv2.namedWindow(encoded_window, cv2.WINDOW_FREERATIO)
    cv2.namedWindow(decoded_window, cv2.WINDOW_FREERATIO)
    img = cv2.imread('img.jpeg')

    # this part generates a qr code from the txt file
    f = open("generate_from_this.txt", 'rb') 
    qr = qrcode.make(f.read()) #cv2.imread('qr.png')
    qr.save("generated_qr.png")
    f.close()
    qr = cv2.imread("generated_qr.png")
    encoded = encoding.encode(img, qr, opacity)

    # this part blurs the image using blur methods that are very effective at denoising
    # its supposed to cause loss of data from the original image
    encoded = cv2.medianBlur(encoded, 7)
    encoded = cv2.GaussianBlur(encoded, (3, 3), 10)

    # decoding and denoising
    decoded = encoding.decode(img, encoded, opacity)
    decoded = encoding.denoise(decoded)

    cv2.imshow(encoded_window, encoded)
    cv2.imshow(decoded_window, decoded)
    
    cv2.imwrite("encoded.jpg", encoded)
    cv2.imwrite("decoded.jpg", decoded)
    
    # read qr
    detector = cv2.QRCodeDetector()
    data = detector.detectAndDecodeMulti(decoded)[1][0]
    print("\n\ndata in your qr code: " + str(data))
    cv2.waitKey(0)
    
    
    

if __name__ == '__main__':
    main()



