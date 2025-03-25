# Define the color bounds for the mask
# YELLOW
Y_LOW_BOUND = (0.084, 0.606, 0.000)
Y_UP_BOUND = (0.171, 1.000, 1.000)
#erode, dilate, close, open
Y_MASK_OPTIONS = (1, 2, 3, 0)

# RED
R_LOW_BOUND = (0.926, 0.629, 0.405)
R_UP_BOUND = (0.069, 1.000, 1.000)

# Dark GREEN
DG_LOW_BOUND = (0.237, 0.319, 0.000)
DG_UP_BOUND = (0.438, 1.000, 1.000)

# Light GREEN
cols = {
        "white": ((.842, .000, .882), (.153, .126, 1.000)),
        "grey": ((0.078, 0.038, 0.687), (0.211, 0.095, 0.851)),
        "black": ((0.022, 0.000, 0.000), (0.245, 0.605, 0.469)),
        "red": ((0.958, 0.431, 0.370), (0.074, 1.000, 1.000)),  # DONE
        "yellow": ((0.107, 0.373, 0.581), (0.201, 1.000, 1.000)),  # DONE
        "light_green": ((0.461, 0.123, 0.745), (0.559, 1.000, 1.000)),
        "dark_green": ((0.308, 0.222, 0.000), (0.460, 1.000, 1.000)),
        # "light_blue": ((), ()),
        "dark_blue": ((0.601, 0.264, 0.405), (0.662, 1.000, 0.965)),
    }


import cv2
import numpy as np

def create_mask(image, color: str):
    color_lower_bound, color_upper_bound = cols[color]
    lower_bound = (np.array(color_lower_bound) * np.array([179, 255, 255])).astype(np.uint8)
    upper_bound = (np.array(color_upper_bound) * np.array([179, 255, 255])).astype(np.uint8)

    # wrap around max value on H channel
    if lower_bound[0] > upper_bound[0]:
        cr_mask = cv2.inRange(image, lower_bound, np.array((179, upper_bound[1], upper_bound[2]), dtype=np.uint8))
        cr_mask += cv2.inRange(image, np.array((0, lower_bound[1], lower_bound[2]), dtype=np.uint8), upper_bound)
    else:
        cr_mask = cv2.inRange(image, lower_bound, upper_bound)

    morph_mask = cr_mask.copy()

    kernels = {
        2: np.ones((2, 2), np.uint8),
        3: np.ones((3, 3), np.uint8),
        5: np.ones((5, 5), np.uint8),
        10: np.ones((10, 10), np.uint8)
    }

    morph_mask = cv2.erode(morph_mask, kernel=kernels[5], iterations=1)
    morph_mask = cv2.dilate(morph_mask, kernel=kernels[5], iterations=1)
    morph_mask = cv2.morphologyEx(morph_mask, op=cv2.MORPH_CLOSE, kernel=kernels[10], iterations=3)
    morph_mask = cv2.morphologyEx(morph_mask, op=cv2.MORPH_OPEN, kernel=kernels[10], iterations=1)

    return cr_mask, morph_mask

def add_shapes(prepared_mask, image):
    contours, _ = cv2.findContours(prepared_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours and identify shapes
    output_image = image.copy()
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
        x, y, w, h = cv2.boundingRect(approx)

        if len(approx) == 3:
            shape_name = "Triangle"
        elif len(approx) == 4:
            shape_name = "Rectangle"
        elif len(approx) > 4:
            shape_name = "Circle"
        else:
            shape_name = "Unknown"

        # Draw bounding box with text
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(output_image, shape_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return output_image



def show_image_and_mask(image, image_mask):
    cv2.imshow("Image", image)
    cv2.imshow("Mask", image_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    image_path = "test.jpg"
    img = cv2.imread(image_path)
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Resize the image to 50% coz it's too big for my screen xd
    img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
    mask, masked_image = create_mask(hsv_image, Y_LOW_BOUND, Y_UP_BOUND, Y_MASK_OPTIONS)
    final_image = add_shapes(mask, img)
    show_image_and_mask(final_image, mask)
