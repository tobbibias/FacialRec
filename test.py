import cv2

def main():
    img = cv2.imread('lfw/Aaron_Eckhart/Aaron_Eckhart_0001.jpg')
    cv2.imshow('test2',img)
    img = cv2.resize(img,(125,125))
    cv2.imshow('test',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()