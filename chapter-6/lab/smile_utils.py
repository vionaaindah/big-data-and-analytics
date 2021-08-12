import cv2
import numpy as np
from scipy.ndimage import zoom


def find_face(frame):
    """
    
    Parameters
    ----------
    
    frame : picture to search for one or more faces
    
    Returns
    -------
    
    gray: a grayscale version of the input frame
    faces: list of lists, containing a quadruple for each detected face: coordinates of the top left corner
           of the location of the face, its width and its height.
    
    Notes
    -----
    
    This function finds faces inside of an image.
    
    """
    cascadePath = "Data/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=10,
            minSize=(100, 100),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
    return gray, faces


def standardize(gray, face, coefficients):
    """
    
    Parameters
    ----------
    
    gray : numpy.ndarray
          it contains the face to be standardized
    face: list 
          it contains the coordinates of the top left corner of the detected face, 
          its height, and its width.
    coefficients: list
          it contains the distortion coefficients 
    
    Returns
    -------
    
    standardized_face: numpy.ndarray
                       the standardized face
    Notes
    -----
    
    This function crops a section of the input image, maps it to a 64x64 pixel square and normalises it between 0 and 1.
    
    """
    (x, y, w, h) = face
    v_cut = int(coefficients[0] * w)
    h_cut = int(coefficients[1] * h)
    original_face = gray[y+v_cut:y+h, 
                      x+h_cut:x-h_cut+w]
    standardized_face = zoom(original_face, (64. / original_face.shape[0], 
                                           64. / original_face.shape[1]))
    standardized_face = standardized_face.astype(np.float32)
    standardized_face /= standardized_face.max()
    return standardized_face

def is_smiling(face, clf):
    """
    
    Parameters
    ----------
    
    face: numpy.ndarray
          the face to be classified
    clf: sklearn classifier (e.g., sklearn.svm.classes.SVC)
          
    Returns
    -------
    
    the predicted label
    
    """
    return clf.predict((face.ravel()).reshape(1, -1))
