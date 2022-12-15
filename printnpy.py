import matplotlib.pylab as plt
import seaborn as sns
import scipy as sp
import dlib

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
print(predictor)