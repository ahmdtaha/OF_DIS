import pyx_flow
import numpy as np
import matplotlib;

import cv2
import flo_utils
import visualize_flo


if __name__ == '__main__':
    print("Hello World")
    flow = pyx_flow.optical_fn(b'./car1.jpg',b'./car2.jpg')
    flo_utils.write_flo(flow,'./py_result.flo')

    img = visualize_flo.computeImg(flow)
    cv2.imshow('display', img)
    k = cv2.waitKey()
