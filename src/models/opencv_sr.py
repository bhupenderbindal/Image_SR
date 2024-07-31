import os
import cv2
import matplotlib.pyplot as plt
from src.visualization.plot_utils import plot_image_grid


def main():
    print(
        os.getcwd(),
    )
    # flag -1 will load with unchanged mode otherwise only BGR mode
    img = cv2.imread("./data/raw/lr.png")
    # historical reasons causes opencv to read in BGR format, hence below
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    # EDSR
    # sr.readModel("./EDSR_x2.pb")
    # sr.setModel("edsr", 2)

    # LapSRN
    sr.readModel("./models/LapSRN_x2.pb")
    sr.setModel("lapsrn", 2)

    result = sr.upsample(img)

    plot_image_grid((img, "original"), (result, "LapSRN_x2"), gridshape=(1, 2))

   


if __name__ == "__main__":
    main()
