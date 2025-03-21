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


import cv2
import numpy as np

def create_mask(image,
                color_lower_bound: tuple[float, float, float], color_upper_bound: tuple[float, float, float],
                dilate_options: tuple[int, int, int, int]):

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_bound = np.array([color_lower_bound[0] * 179, color_lower_bound[1] * 255, color_lower_bound[2] * 255], dtype=np.uint8)
    upper_bound = np.array([color_upper_bound[0] * 179, color_upper_bound[1] * 255, color_upper_bound[2] * 255], dtype=np.uint8)

    cr_mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

    # Apply mask to the original image
    masked_rgb_image = cv2.bitwise_and(image, image, mask=cr_mask)
    kernel = np.ones((5, 5), np.uint8)
    # create mask
    cr_mask = cv2.erode(cr_mask, kernel, iterations=dilate_options[0])
    # cr_mask = cv2.dilate(cr_mask, kernel, iterations=dilate_options[1])
    cr_mask = cv2.morphologyEx(cr_mask, cv2.MORPH_CLOSE, kernel, iterations=dilate_options[2])
    cr_mask = cv2.morphologyEx(cr_mask, cv2.MORPH_CLOSE, kernel, iterations=dilate_options[2])
    cr_mask = cv2.morphologyEx(cr_mask, cv2.MORPH_OPEN, kernel, iterations=dilate_options[3])

    return cr_mask, masked_rgb_image

def add_shapes(prepared_mask, image):
    contours, _ = cv2.findContours(prepared_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours and identify shapes
    output_image = image.copy()
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        x, y, w, h = cv2.boundingRect(approx)

        if len(approx) == 3:
            shape_name = "Triangle"
        elif len(approx) == 4:
            shape_name = "Rectangle"
        elif len(approx) > 4:
            shape_name = "Circle"
        else:
            shape_name = "Unknown"

        # Draw bounding box
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Put text label
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
    # Resize the image to 50% coz it's too big for my screen xd
    img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
    mask, masked_image = create_mask(img, Y_LOW_BOUND, Y_UP_BOUND, Y_MASK_OPTIONS)
    final_image = add_shapes(mask, img)
    show_image_and_mask(final_image, mask)
