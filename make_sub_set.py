import os
import sys
import cv2
import random
# this very script only moves the folders and their content to a new dir called "small_Lfw"


def containsThree(input_path, person):
    i = 0
    for img in os.listdir(f'{input_path}/{person}'):
        i += 1
    if i >= 3:
        return True
    else:
        return False


def main(input_path, output_path):
    os.mkdir(output_path)
    people = 0
    filenames = random.sample(os.listdir(input_path), 800)

    for person in filenames:
        if person.startswith('.'):
            pass
        else:
            if containsThree(input_path, person) and not people > 100:
                print(['+'])
                try:
                    os.mkdir(f'{output_path}/{person}')
                except Exception:
                    pass
                for img in os.listdir(f'{input_path}/{person}/'):
                    if img.startswith('.'):
                        pass
                    else:
                        image = cv2.imread(f'{input_path}/{person}/{img}')
                        cv2.imwrite(f'{output_path}/{person}/{img}', image)
                people += 1
            else:
                print(['-'])
                pass


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
