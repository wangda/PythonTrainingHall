from time import sleep

import cv2


def read_image():

    # 支持的图片格式:
    # Windows位图: .bmp, .dib,
    # JPEG/JPEG2000: .png, .jpg, .jpeg, .jp2,
    # 便携式图像格式: .pbm, .pgm, .ppm, .pxm, .pnm
    # .sr, .ras
    # .webp
    # 读取一个图片, 返回图片, 数据类型为: cv2.mat_wrapper.Mat 或 ndarray(NumPy的多维数组)
    img = cv2.imread("lena.png")
    return img

def save_image(img):
    # 保存图片, 参数: 图片, 文件名, 图片格式
    cv2.imwrite("lena_copy.bmp", img)

img = read_image()
# 创建一个窗口
cv2.namedWindow("Hello")
print(img)

# 显示图片
cv2.imshow("Press q Exit", img)

# 等待用户输入, 按下 q 键退出
cv2.waitKey(3000) == ord('q')

# 释放窗口
cv2.destroyAllWindows()

save_image(img)
