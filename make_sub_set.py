import os
import sys
import cv2

# this very script only moves the folders and their content to a new dir called "small_Lfw"

def containsThree(indir, person):
    i = 0
    for img in os.listdir(f'{indir}/{person}'):
        i += 1
    if i >= 3:
        return True
    else:
        return False

def main(indir, outdir):
    os.mkdir(outdir)
    i = 0
    for person in os.listdir(indir):
        if person.startswith('.'):
            pass
        else:
            if containsThree(indir, person) and not i > 100:
                print(['+'])
                try:
                    os.mkdir(f'{outdir}/{person}')
                except Exception:
                    pass
                for img in os.listdir(f'{indir}/{person}/'):
                    if img.startswith('.'):
                        pass
                    else:
                        image = cv2.imread(f'{indir}/{person}/{img}')
                        cv2.imwrite(f'{outdir}/{person}/{img}', image)
                i += 1
            else:
                print(['-'])
                pass

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
