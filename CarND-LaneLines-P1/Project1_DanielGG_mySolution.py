import numpy as np
import cv2

# ===================================input and output====================================
_input_video = cv2.VideoCapture('solidWhiteRight.mp4')
#_input_video = cv2.VideoCapture('solidYellowLeft.mp4')

_codec = cv2.VideoWriter_fourcc(*'XVID')
_output_video = cv2.VideoWriter('output.avi', _codec, 20.0, (640,480))

# ======================================variables========================================
kernel_size = 5
low_threshold = 50
high_threshold = 150
rho = 2                         # distance resolution in pixels of the Hough grid - Original: 1
theta = np.pi/180               # angular resolution in radians of the Hough grid
threshold = 15                  # minimum number of votes (intersections in Hough grid cell) - Original: 1
min_line_length = 40            # minimum number of pixels making up a line - Original: 1
max_line_gap = 20               # maximum gap in pixels between connectable line segments - Original: 1

_mask_points = [60, 540, 450, 315, 495, 315, 910, 540]

# ======================================funciones========================================
def canny_edge_det(_img_):  #generates the final edges (includes grey, Blurr, and Canny)
    first_step_grey = cv2.cvtColor(_img_, cv2.COLOR_BGR2GRAY)
    second_step_blur = cv2.GaussianBlur(first_step_grey, (kernel_size, kernel_size), 0)
    third_step_canny = cv2.Canny(second_step_blur, low_threshold, high_threshold)
    return third_step_canny

def generate_ROI(canny, x, y, poly_points): # generates region of interest, returns image
    _mask = np.zeros_like(canny)
    ignore_mask_color = 255
    _mask_shape = (x, y)
    _mask_vertices = np.array(
        [[(poly_points[0], poly_points[1]), (poly_points[2], poly_points[3]), (poly_points[4], poly_points[5]), (poly_points[6], poly_points[7])]],
        dtype=np.int32)
    cv2.fillPoly(_mask, _mask_vertices, ignore_mask_color)
    _masked_edges = cv2.bitwise_and(edges, _mask)
    return _masked_edges

# ======================================ejecucion========================================
while _input_video.isOpened():
    successful_frame, original = _input_video.read()
    if successful_frame == True:    # si es que se pudo leer el video....

        edges = canny_edge_det(original)
        _masked_edges = generate_ROI(edges, 540, 960, _mask_points)

        # copia frame original y empalma lineas (mas adelante)
        line_image = np.copy(original) * 0   # algo esta mal aqui

        # Output "lines" is an array containing endpoints of detected line segments
        _lines = cv2.HoughLinesP(_masked_edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

        # Iterate over the output "lines" and draw lines on a blank image
        for line in _lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Draw the lines on the edge image
        color_edges = np.dstack((edges, edges, edges))
        lines_edges = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)

        # Para salvar video
        _output_video.write(lines_edges) # algo falta

        # para mostrar "videos" en pantalla
        cv2.imshow("imgOriginal", original)
        cv2.imshow("imgCanny", lines_edges)

        if cv2.waitKey(25) & 0xFF == ord('q'):  # tiempo entre frames y tecla para terminar programa (esq y taches no funcionan)
            break
    else:
        break

# cerrar streams (asumo hay que cerrar output primero)

_input_video.release()
_output_video.release()