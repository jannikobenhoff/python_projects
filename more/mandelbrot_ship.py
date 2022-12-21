import numpy as np
from PIL import Image

if __name__ == "__main__":
    # 800 x 450 pixel
    x_dim = 800
    y_dim = 450
    main_color = [255, 7, 15]
    im = np.ndarray([x_dim, y_dim, 3]).astype('uint8')
    xindex = 0
    yindex = 0
    for x in np.linspace(-1.8, -1.7, x_dim):
        for y in np.linspace(-0.08, 0.01, y_dim):
            zx, zy = x, y
            iteration = 0
            max_iteration = 64

            while zx*zx + zy*zy < 4 and iteration < max_iteration:
                xtemp = zx * zx - zy * zy + x
                zy = abs(2 * zx * zy) + y
                zx = xtemp
                iteration = iteration + 1

            if iteration == max_iteration:
                im[xindex, yindex] = main_color
            else:
                im[xindex, yindex] = [main_color[0], main_color[1] * iteration, main_color[2] * iteration]
            yindex = yindex + 1
            if yindex >= y_dim:
                yindex = 0
        xindex = xindex +1
        if xindex >= x_dim:
            xindex = 0

    ship = Image.fromarray(im)
    ship = ship.convert('RGB')
    ship.save("fractal.png")
    ship = Image.open("fractal.png")
    ship = ship.rotate(-90, expand=1)
    ship = ship.transpose(Image.FLIP_LEFT_RIGHT)
    #ship.show()
    ship.save("fractal.png")


