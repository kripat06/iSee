import numpy as np

cardW = 57
cardH = 89
cornerXmin = 2.5
cornerXmax = 10.5
cornerYmin = 3.5
cornerYmax = 24

S_cornerXmin = 2.5
S_cornerXmax = 8.9
S_cornerYmin = 3.2
S_cornerYmax = 23

C_cornerXmin = 2.5
C_cornerXmax = 9.45
C_cornerYmin = 3.5
C_cornerYmax = 23

D_cornerXmin = 2.5
D_cornerXmax = 9.45
D_cornerYmin = 3.5
D_cornerYmax = 23

H_cornerXmin = 2.5
H_cornerXmax = 9.45
H_cornerYmin = 3.5
H_cornerYmax = 23


# We convert the measures from mm to pixels: multiply by an arbitrary factor 'zoom'
# You shouldn't need to change this
zoom = 4
cardW *= zoom
cardH *= zoom
cornerXmin = int(cornerXmin * zoom)
cornerXmax = int(cornerXmax * zoom)
cornerYmin = int(cornerYmin * zoom)
cornerYmax = int(cornerYmax * zoom)

C_cornerXmin = int(C_cornerXmin * zoom)
C_cornerXmax = int(C_cornerXmax * zoom)
C_cornerYmin = int(C_cornerYmin * zoom)
C_cornerYmax = int(C_cornerYmax * zoom)

D_cornerXmin = int(D_cornerXmin * zoom)
D_cornerXmax = int(D_cornerXmax * zoom)
D_cornerYmin = int(D_cornerYmin * zoom)
D_cornerYmax = int(D_cornerYmax * zoom)

H_cornerXmin = int(H_cornerXmin * zoom)
H_cornerXmax = int(H_cornerXmax * zoom)
H_cornerYmin = int(H_cornerYmin * zoom)
H_cornerYmax = int(H_cornerYmax * zoom)

S_cornerXmin = int(S_cornerXmin * zoom)
S_cornerXmax = int(S_cornerXmax * zoom)
S_cornerYmin = int(S_cornerYmin * zoom)
S_cornerYmax = int(S_cornerYmax * zoom)

S_refCornerHL = np.array(
    [[S_cornerXmin, S_cornerYmin], [S_cornerXmax, S_cornerYmin], [S_cornerXmax, S_cornerYmax],
     [S_cornerXmin, S_cornerYmax]],
    dtype=np.float32)
S_refCornerLR = np.array([[cardW - S_cornerXmax, cardH - S_cornerYmax], [cardW - S_cornerXmin, cardH - S_cornerYmax],
                          [cardW - S_cornerXmin, cardH - S_cornerYmin], [cardW - S_cornerXmax, cardH - S_cornerYmin]],
                         dtype=np.float32)

H_refCornerHL = np.array(
    [[H_cornerXmin, H_cornerYmin], [H_cornerXmax, H_cornerYmin], [H_cornerXmax, H_cornerYmax],
     [H_cornerXmin, H_cornerYmax]],
    dtype=np.float32)
H_refCornerLR = np.array([[cardW - H_cornerXmax, cardH - H_cornerYmax], [cardW - H_cornerXmin, cardH - H_cornerYmax],
                          [cardW - H_cornerXmin, cardH - H_cornerYmin], [cardW - H_cornerXmax, cardH - H_cornerYmin]],
                         dtype=np.float32)

D_refCornerHL = np.array(
    [[D_cornerXmin, D_cornerYmin], [D_cornerXmax, D_cornerYmin], [D_cornerXmax, D_cornerYmax],
     [D_cornerXmin, D_cornerYmax]],
    dtype=np.float32)
D_refCornerLR = np.array([[cardW - D_cornerXmax, cardH - D_cornerYmax], [cardW - D_cornerXmin, cardH - D_cornerYmax],
                          [cardW - D_cornerXmin, cardH - D_cornerYmin], [cardW - D_cornerXmax, cardH - D_cornerYmin]],
                         dtype=np.float32)

C_refCornerHL = np.array(
    [[C_cornerXmin, C_cornerYmin], [C_cornerXmax, C_cornerYmin], [C_cornerXmax, C_cornerYmax],
     [C_cornerXmin, C_cornerYmax]],
    dtype=np.float32)
C_refCornerLR = np.array([[cardW - C_cornerXmax, cardH - C_cornerYmax], [cardW - C_cornerXmin, cardH - C_cornerYmax],
                          [cardW - C_cornerXmin, cardH - C_cornerYmin], [cardW - C_cornerXmax, cardH - C_cornerYmin]],
                         dtype=np.float32)

refCard = np.array([[0, 0], [cardW, 0], [cardW, cardH], [0, cardH]], dtype=np.float32)
refCardRot = np.array([[cardW, 0], [cardW, cardH], [0, cardH], [0, 0]], dtype=np.float32)
refCornerHL = np.array(
    [[cornerXmin, cornerYmin], [cornerXmax, cornerYmin], [cornerXmax, cornerYmax], [cornerXmin, cornerYmax]],
    dtype=np.float32)
refCornerLR = np.array([[cardW - cornerXmax, cardH - cornerYmax], [cardW - cornerXmin, cardH - cornerYmax],
                        [cardW - cornerXmin, cardH - cornerYmin], [cardW - cornerXmax, cardH - cornerYmin]],
                       dtype=np.float32)
refCorners = np.array([refCornerHL, refCornerLR])

# imgW,imgH: dimensions of the generated dataset bg_images
imgW = 720
imgH = 720
