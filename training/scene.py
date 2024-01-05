import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import random
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import pickle
from glob import glob
import imgaug as ia
from imgaug import augmenters as iaa
from shapely.geometry import Polygon
from ic.img.image_support import *
from ic.support.create_voc_xml import *
from ic.support.measurements import *
from ic.scene.card_scene_maker import *
from ic.scene.BBA import *


class Scene:
    def __init__(self, bg, img1=None, class1=None, hulla1=None, hullb1=None, img2=None, class2=None, hulla2=None,
                 hullb2=None, img3=None, class3=None,
                 hulla3=None, hullb3=None, img4=None, class4=None, hulla4=None, hullb4=None, img5=None, class5=None,
                 hulla5=None, hullb5=None, img6=None,
                 class6=None, hulla6=None, hullb6=None, img7=None, class7=None, hulla7=None, hullb7=None, img8=None,
                 class8=None, hulla8=None, hullb8=None,
                 img9=None, class9=None, hulla9=None, hullb9=None):
        if img9 is not None:
            self.create9CardsScene(bg, img1, class1, hulla1, hullb1, img2, class2, hulla2, hullb2, img3, class3, hulla3,
                                   hullb3, img4, class4, hulla4,
                                   hullb4, img5, class5, hulla5,
                                   hullb5, img6, class6, hulla6,
                                   hullb6, img7, class7, hulla7,
                                   hullb7, img8, class8, hulla8,
                                   hullb8, img9, class9, hulla9,
                                   hullb9)
        elif img6 is not None:
            self.create6CardsScene(bg, img1, class1, hulla1, hullb1, img2, class2, hulla2, hullb2, img3, class3, hulla3,
                                   hullb3, img4, class4, hulla4,
                                   hullb4, img5, class5, hulla5,
                                   hullb5, img6, class6, hulla6,
                                   hullb6)
        elif img5 is not None:
            self.create5CardsScene(bg, img1, class1, hulla1, hullb1, img2, class2, hulla2, hullb2, img3, class3, hulla3,
                                   hullb3, img4, class4, hulla4,
                                   hullb4, img5, class5, hulla5,
                                   hullb5)
        elif img4 is not None:
            self.create4CardsScene(bg, img1, class1, hulla1, hullb1, img2, class2, hulla2, hullb2, img3, class3, hulla3,
                                   hullb3, img4, class4, hulla4,
                                   hullb4)
        elif img3 is not None:
            self.create3CardsScene(bg, img1, class1, hulla1, hullb1, img2, class2, hulla2, hullb2, img3, class3, hulla3,
                                   hullb3)
        elif img1 is not None:
            self.create1CardScene(bg, img1, class1, hulla1, hullb1)
        elif img2 is not None:
            self.create3CardsScene(bg, img1, class1, hulla1, hullb1, img2, class2, hulla2, hullb2)
        else:
            self.create2CardsScene(bg, img1, class1, hulla1, hullb1, img2, class2, hulla2, hullb2)

    def generate1CardData(self, bg, img1, class1, bboxa1, bboxb1):

        # cords_to_kps converts the bounding box points into the keypoint format which imgaug requires
        keya1 = bbox_to_key(bboxa1)
        keyb1 = bbox_to_key(bboxb1)

        # Creates empty array with the dimensions of the input card image
        self.img1 = np.zeros((imgH, imgW, 4), dtype=np.uint8)

        # Uses array slicing to move just the card into a set position inside the larger img1 array
        self.img1[decalY:decalY + cardH, decalX:decalX + cardW, :] = img1

        # Uses the imgaug library to assign random rotation to the card
        self.img1, self.lkps1, self.bbs1 = augment(self.img1, [cardKP, keya1, keyb1], transform_1card)

        # Acsesses the bounding box and label of the card
        self.class1 = class1
        self.listbba = [BBA(self.bbs1[0], class1)]

        # Randomly orients the backgroud
        self.bg = scaleBg.augment_image(bg)

        # Extracts the around the card as a mask using the alpha channel
        mask1 = self.img1[:, :, 3]

        # Applies the mask to all layers (red, gree, and blue)
        self.mask1 = np.stack([mask1] * 3, -1)

        # Blends both the individual card and the backgound using the mask
        self.final = np.where(self.mask1, self.img1[:, :, 0:3], self.bg)

    def create2CardsScene(self, bg, img1, class1, hulla1, hullb1, img2, class2, hulla2, hullb2):
        kpsa1 = hull_to_kps(hulla1)
        kpsb1 = hull_to_kps(hullb1)
        kpsa2 = hull_to_kps(hulla2)
        kpsb2 = hull_to_kps(hullb2)

        # Randomly transform 1st card
        self.img1 = np.zeros((imgH, imgW, 4), dtype=np.uint8)
        self.img1[decalY:decalY + cardH, decalX:decalX + cardW, :] = img1
        self.img1, self.lkps1, self.bbs1 = augment(self.img1, [cardKP, kpsa1, kpsb1], transform_1card)

        # Randomly transform 2nd card. We want that card 2 does not partially cover a corner of 1 card.
        # If so, we apply a new random transform to card 2
        while True:
            self.listbba = []
            self.img2 = np.zeros((imgH, imgW, 4), dtype=np.uint8)
            self.img2[decalY:decalY + cardH, decalX:decalX + cardW, :] = img2
            self.img2, self.lkps2, self.bbs2 = augment(self.img2, [cardKP, kpsa2, kpsb2], transform_1card)

            # mainPoly2: shapely polygon of card 2
            mainPoly2 = kps_to_polygon(self.lkps2[0].keypoints[0:4])
            invalid = False
            intersect_ratio = 0.1
            for i in range(1, 3):
                # smallPoly1: shapely polygon of one of the hull of card 1
                smallPoly1 = kps_to_polygon(self.lkps1[i].keypoints[:])
                a = smallPoly1.area
                # We calculate area of the intersection of card 1 corner with card 2
                intersect = mainPoly2.intersection(smallPoly1)
                ai = intersect.area
                # If intersection area is small enough, we accept card 2
                if (a - ai) / a > 1 - intersect_ratio:
                    self.listbba.append(BBA(self.bbs1[i - 1], class1))
                # If intersectio area is not small, but also not big enough, we want apply new transform to card 2
                elif (a - ai) / a > intersect_ratio:
                    invalid = True
                    break

            if not invalid: break

        self.class1 = class1
        self.class2 = class2
        for bb in self.bbs2:
            self.listbba.append(BBA(bb, class2))
        # Construct final image of the scene by superimposing: bg, img1 and img2
        self.bg = scaleBg.augment_image(bg)
        mask1 = self.img1[:, :, 3]
        self.mask1 = np.stack([mask1] * 3, -1)
        self.final = np.where(self.mask1, self.img1[:, :, 0:3], self.bg)
        mask2 = self.img2[:, :, 3]
        self.mask2 = np.stack([mask2] * 3, -1)
        self.final = np.where(self.mask2, self.img2[:, :, 0:3], self.final)

    def create3CardsScene(self, bg, img1, class1, hulla1, hullb1, img2, class2, hulla2, hullb2, img3, class3, hulla3,
                          hullb3):

        kpsa1 = hull_to_kps(hulla1, decalX3, decalY3)
        kpsb1 = hull_to_kps(hullb1, decalX3, decalY3)
        kpsa2 = hull_to_kps(hulla2, decalX3, decalY3)
        kpsb2 = hull_to_kps(hullb2, decalX3, decalY3)
        kpsa3 = hull_to_kps(hulla3, decalX3, decalY3)
        kpsb3 = hull_to_kps(hullb3, decalX3, decalY3)
        self.img3 = np.zeros((imgH, imgW, 4), dtype=np.uint8)
        self.img3[decalY3:decalY3 + cardH, decalX3:decalX3 + cardW, :] = img3
        self.img3, self.lkps3, self.bbs3 = augment(self.img3, [cardKP, kpsa3, kpsb3], trans_rot1)
        self.img2 = np.zeros((imgH, imgW, 4), dtype=np.uint8)
        self.img2[decalY3:decalY3 + cardH, decalX3:decalX3 + cardW, :] = img2
        self.img2, self.lkps2, self.bbs2 = augment(self.img2, [cardKP, kpsa2, kpsb2], trans_rot2)
        self.img1 = np.zeros((imgH, imgW, 4), dtype=np.uint8)
        self.img1[decalY3:decalY3 + cardH, decalX3:decalX3 + cardW, :] = img1

        while True:
            det_transform_3cards = transform_3cards.to_deterministic()
            _img3, _lkps3, self.bbs3 = augment(self.img3, self.lkps3, det_transform_3cards, False)
            if _img3 is None: continue
            _img2, _lkps2, self.bbs2 = augment(self.img2, self.lkps2, det_transform_3cards, False)
            if _img2 is None: continue
            _img1, self.lkps1, self.bbs1 = augment(self.img1, [cardKP, kpsa1, kpsb1], det_transform_3cards, False)
            if _img1 is None: continue
            break
        self.img3 = _img3
        self.lkps3 = _lkps3
        self.img2 = _img2
        self.lkps2 = _lkps2
        self.img1 = _img1

        self.class1 = class1
        self.class2 = class2
        self.class3 = class3
        self.listbba = [BBA(self.bbs1[0], class1), BBA(self.bbs2[0], class2), BBA(self.bbs3[0], class3),
                        BBA(self.bbs3[1], class3)]

        # Construct final image of the scene by superimposing: bg, img1, img2 and img3
        self.bg = scaleBg.augment_image(bg)
        mask1 = self.img1[:, :, 3]
        self.mask1 = np.stack([mask1] * 3, -1)
        self.final = np.where(self.mask1, self.img1[:, :, 0:3], self.bg)
        mask2 = self.img2[:, :, 3]
        self.mask2 = np.stack([mask2] * 3, -1)
        self.final = np.where(self.mask2, self.img2[:, :, 0:3], self.final)
        mask3 = self.img3[:, :, 3]
        self.mask3 = np.stack([mask3] * 3, -1)
        self.final = np.where(self.mask3, self.img3[:, :, 0:3], self.final)

    def create4CardsScene(self, bg, img1, class1, hulla1, hullb1, img2, class2, hulla2, hullb2, img3, class3, hulla3,
                          hullb3, img4, class4, hulla4, hullb4):

        kpsa1 = hull_to_kps(hulla1, decalX3, decalY3)
        kpsb1 = hull_to_kps(hullb1, decalX3, decalY3)
        kpsa2 = hull_to_kps(hulla2, decalX3, decalY3)
        kpsb2 = hull_to_kps(hullb2, decalX3, decalY3)
        kpsa3 = hull_to_kps(hulla3, decalX3, decalY3)
        kpsb3 = hull_to_kps(hullb3, decalX3, decalY3)
        kpsa4 = hull_to_kps(hulla4, decalX3, decalY3)
        kpsb4 = hull_to_kps(hullb4, decalX3, decalY3)

        self.img4 = np.zeros((imgH, imgW, 4), dtype=np.uint8)
        self.img4[decalY3:decalY3 + cardH, decalX3:decalX3 + cardW, :] = img4
        self.img4, self.lkps4, self.bbs4 = augment(self.img4, [cardKP, kpsa4, kpsb4], trans_rot1)
        self.img3 = np.zeros((imgH, imgW, 4), dtype=np.uint8)
        self.img3[decalY3:decalY3 + cardH, decalX3:decalX3 + cardW, :] = img3
        self.img3, self.lkps3, self.bbs3 = augment(self.img3, [cardKP, kpsa3, kpsb3], trans_rot2)
        self.img2 = np.zeros((imgH, imgW, 4), dtype=np.uint8)
        self.img2[decalY3:decalY3 + cardH, decalX3:decalX3 + cardW, :] = img2
        self.img2, self.lkps2, self.bbs2 = augment(self.img2, [cardKP, kpsa2, kpsb2], trans_rot3)
        self.img1 = np.zeros((imgH, imgW, 4), dtype=np.uint8)
        self.img1[decalY3:decalY3 + cardH, decalX3:decalX3 + cardW, :] = img1

        while True:
            det_transform_3cards = transform_3cards.to_deterministic()
            _img4, _lkps4, self.bbs4 = augment(self.img4, self.lkps4, det_transform_3cards, False)
            if _img4 is None: continue
            _img3, _lkps3, self.bbs3 = augment(self.img3, self.lkps3, det_transform_3cards, False)
            if _img3 is None: continue
            _img2, _lkps2, self.bbs2 = augment(self.img2, self.lkps2, det_transform_3cards, False)
            if _img2 is None: continue
            _img1, self.lkps1, self.bbs1 = augment(self.img1, [cardKP, kpsa1, kpsb1], det_transform_3cards, False)
            if _img1 is None: continue
            break
        self.img4 = _img4
        self.lkps4 = _lkps4
        self.img3 = _img3
        self.lkps3 = _lkps3
        self.img2 = _img2
        self.lkps2 = _lkps2
        self.img1 = _img1

        self.class1 = class1
        self.class2 = class2
        self.class3 = class3
        self.class4 = class4
        self.listbba = [BBA(self.bbs1[0], class1), BBA(self.bbs2[0], class2), BBA(self.bbs3[0], class3),
                        BBA(self.bbs3[0], class3), BBA(self.bbs4[0], class4)]

        # Construct final image of the scene by superimposing: bg, img1, img2 and img3
        self.bg = scaleBg.augment_image(bg)
        mask1 = self.img1[:, :, 3]
        self.mask1 = np.stack([mask1] * 3, -1)
        self.final = np.where(self.mask1, self.img1[:, :, 0:3], self.bg)
        mask2 = self.img2[:, :, 3]
        self.mask2 = np.stack([mask2] * 3, -1)
        self.final = np.where(self.mask2, self.img2[:, :, 0:3], self.final)
        mask3 = self.img3[:, :, 3]
        self.mask3 = np.stack([mask3] * 3, -1)
        self.final = np.where(self.mask3, self.img3[:, :, 0:3], self.final)
        mask4 = self.img4[:, :, 3]
        self.mask4 = np.stack([mask4] * 3, -1)
        self.final = np.where(self.mask4, self.img4[:, :, 0:3], self.final)

    def create5CardsScene(self, bg, img1, class1, hulla1, hullb1, img2, class2, hulla2, hullb2, img3, class3, hulla3,
                          hullb3, img4, class4, hulla4, hullb4, img5, class5, hulla5, hullb5):

        kpsa1 = hull_to_kps(hulla1, decalX3, decalY3)
        kpsb1 = hull_to_kps(hullb1, decalX3, decalY3)
        kpsa2 = hull_to_kps(hulla2, decalX3, decalY3)
        kpsb2 = hull_to_kps(hullb2, decalX3, decalY3)
        kpsa3 = hull_to_kps(hulla3, decalX3, decalY3)
        kpsb3 = hull_to_kps(hullb3, decalX3, decalY3)
        kpsa4 = hull_to_kps(hulla4, decalX3, decalY3)
        kpsb4 = hull_to_kps(hullb4, decalX3, decalY3)
        kpsa5 = hull_to_kps(hulla5, decalX3, decalY3)
        kpsb5 = hull_to_kps(hullb5, decalX3, decalY3)

        self.img5 = np.zeros((imgH, imgW, 4), dtype=np.uint8)
        self.img5[decalY3:decalY3 + cardH, decalX3:decalX3 + cardW, :] = img5
        self.img5, self.lkps5, self.bbs5 = augment(self.img5, [cardKP, kpsa5, kpsb5], trans_rot4)
        self.img4 = np.zeros((imgH, imgW, 4), dtype=np.uint8)
        self.img4[decalY3:decalY3 + cardH, decalX3:decalX3 + cardW, :] = img4
        self.img4, self.lkps4, self.bbs4 = augment(self.img4, [cardKP, kpsa4, kpsb4], trans_rot1)
        self.img3 = np.zeros((imgH, imgW, 4), dtype=np.uint8)
        self.img3[decalY3:decalY3 + cardH, decalX3:decalX3 + cardW, :] = img3
        self.img3, self.lkps3, self.bbs3 = augment(self.img3, [cardKP, kpsa3, kpsb3], trans_rot2)
        self.img2 = np.zeros((imgH, imgW, 4), dtype=np.uint8)
        self.img2[decalY3:decalY3 + cardH, decalX3:decalX3 + cardW, :] = img2
        self.img2, self.lkps2, self.bbs2 = augment(self.img2, [cardKP, kpsa2, kpsb2], trans_rot3)
        self.img1 = np.zeros((imgH, imgW, 4), dtype=np.uint8)
        self.img1[decalY3:decalY3 + cardH, decalX3:decalX3 + cardW, :] = img1

        while True:
            det_transform_3cards = transform_3cards.to_deterministic()
            _img5, _lkps5, self.bbs5 = augment(self.img5, self.lkps5, det_transform_3cards, False)
            if _img5 is None: continue
            _img4, _lkps4, self.bbs4 = augment(self.img4, self.lkps4, det_transform_3cards, False)
            if _img4 is None: continue
            _img3, _lkps3, self.bbs3 = augment(self.img3, self.lkps3, det_transform_3cards, False)
            if _img3 is None: continue
            _img2, _lkps2, self.bbs2 = augment(self.img2, self.lkps2, det_transform_3cards, False)
            if _img2 is None: continue
            _img1, self.lkps1, self.bbs1 = augment(self.img1, [cardKP, kpsa1, kpsb1], det_transform_3cards, False)
            if _img1 is None: continue
            break
        self.img5 = _img5
        self.lkps5 = _lkps5
        self.img4 = _img4
        self.lkps4 = _lkps4
        self.img3 = _img3
        self.lkps3 = _lkps3
        self.img2 = _img2
        self.lkps2 = _lkps2
        self.img1 = _img1

        self.class1 = class1
        self.class2 = class2
        self.class3 = class3
        self.class4 = class4
        self.class5 = class5
        self.listbba = [BBA(self.bbs1[0], class1), BBA(self.bbs2[0], class2), BBA(self.bbs3[0], class3),
                        BBA(self.bbs4[0], class4), BBA(self.bbs5[0], class5)]

        # Construct final image of the scene by superimposing: bg, img1, img2 and img3
        self.bg = scaleBg.augment_image(bg)
        mask1 = self.img1[:, :, 3]
        self.mask1 = np.stack([mask1] * 3, -1)
        self.final = np.where(self.mask1, self.img1[:, :, 0:3], self.bg)
        mask2 = self.img2[:, :, 3]
        self.mask2 = np.stack([mask2] * 3, -1)
        self.final = np.where(self.mask2, self.img2[:, :, 0:3], self.final)
        mask3 = self.img3[:, :, 3]
        self.mask3 = np.stack([mask3] * 3, -1)
        self.final = np.where(self.mask3, self.img3[:, :, 0:3], self.final)
        mask4 = self.img4[:, :, 3]
        self.mask4 = np.stack([mask4] * 3, -1)
        self.final = np.where(self.mask4, self.img4[:, :, 0:3], self.final)
        mask5 = self.img5[:, :, 3]
        self.mask5 = np.stack([mask5] * 3, -1)
        self.final = np.where(self.mask5, self.img5[:, :, 0:3], self.final)

    def create6CardsScene(self, bg, img1, class1, hulla1, hullb1, img2, class2, hulla2, hullb2, img3, class3, hulla3,
                          hullb3, img4, class4, hulla4, hullb4, img5, class5, hulla5, hullb5, img6, class6, hulla6,
                          hullb6):

        kpsa1 = hull_to_kps(hulla1, decalX3, decalY3)
        kpsb1 = hull_to_kps(hullb1, decalX3, decalY3)
        kpsa2 = hull_to_kps(hulla2, decalX3, decalY3)
        kpsb2 = hull_to_kps(hullb2, decalX3, decalY3)
        kpsa3 = hull_to_kps(hulla3, decalX3, decalY3)
        kpsb3 = hull_to_kps(hullb3, decalX3, decalY3)
        kpsa4 = hull_to_kps(hulla4, decalX3, decalY3)
        kpsb4 = hull_to_kps(hullb4, decalX3, decalY3)
        kpsa5 = hull_to_kps(hulla5, decalX3, decalY3)
        kpsb5 = hull_to_kps(hullb5, decalX3, decalY3)
        kpsa6 = hull_to_kps(hulla6, decalX3, decalY3)
        kpsb6 = hull_to_kps(hullb6, decalX3, decalY3)

        self.img6 = np.zeros((imgH, imgW, 4), dtype=np.uint8)
        self.img6[decalY3:decalY3 + cardH, decalX3:decalX3 + cardW, :] = img6
        self.img6, self.lkps6, self.bbs6 = augment(self.img6, [cardKP, kpsa6, kpsb6], trans_rot5)
        self.img5 = np.zeros((imgH, imgW, 4), dtype=np.uint8)
        self.img5[decalY3:decalY3 + cardH, decalX3:decalX3 + cardW, :] = img5
        self.img5, self.lkps5, self.bbs5 = augment(self.img5, [cardKP, kpsa5, kpsb5], trans_rot4)
        self.img4 = np.zeros((imgH, imgW, 4), dtype=np.uint8)
        self.img4[decalY3:decalY3 + cardH, decalX3:decalX3 + cardW, :] = img4
        self.img4, self.lkps4, self.bbs4 = augment(self.img4, [cardKP, kpsa4, kpsb4], trans_rot1)
        self.img3 = np.zeros((imgH, imgW, 4), dtype=np.uint8)
        self.img3[decalY3:decalY3 + cardH, decalX3:decalX3 + cardW, :] = img3
        self.img3, self.lkps3, self.bbs3 = augment(self.img3, [cardKP, kpsa3, kpsb3], trans_rot2)
        self.img2 = np.zeros((imgH, imgW, 4), dtype=np.uint8)
        self.img2[decalY3:decalY3 + cardH, decalX3:decalX3 + cardW, :] = img2
        self.img2, self.lkps2, self.bbs2 = augment(self.img2, [cardKP, kpsa2, kpsb2], trans_rot3)
        self.img1 = np.zeros((imgH, imgW, 4), dtype=np.uint8)
        self.img1[decalY3:decalY3 + cardH, decalX3:decalX3 + cardW, :] = img1

        while True:
            det_transform_3cards = transform_3cards.to_deterministic()
            _img6, _lkps6, self.bbs6 = augment(self.img6, self.lkps6, det_transform_3cards, False)
            if _img6 is None: continue
            _img5, _lkps5, self.bbs5 = augment(self.img5, self.lkps5, det_transform_3cards, False)
            if _img5 is None: continue
            _img4, _lkps4, self.bbs4 = augment(self.img4, self.lkps4, det_transform_3cards, False)
            if _img4 is None: continue
            _img3, _lkps3, self.bbs3 = augment(self.img3, self.lkps3, det_transform_3cards, False)
            if _img3 is None: continue
            _img2, _lkps2, self.bbs2 = augment(self.img2, self.lkps2, det_transform_3cards, False)
            if _img2 is None: continue
            _img1, self.lkps1, self.bbs1 = augment(self.img1, [cardKP, kpsa1, kpsb1], det_transform_3cards, False)
            if _img1 is None: continue
            break

        self.img6 = _img6
        self.lkps6 = _lkps6
        self.img5 = _img5
        self.lkps5 = _lkps5
        self.img4 = _img4
        self.lkps4 = _lkps4
        self.img3 = _img3
        self.lkps3 = _lkps3
        self.img2 = _img2
        self.lkps2 = _lkps2
        self.img1 = _img1

        self.class1 = class1
        self.class2 = class2
        self.class3 = class3
        self.class4 = class4
        self.class5 = class5
        self.class6 = class6
        self.listbba = [BBA(self.bbs1[0], class1), BBA(self.bbs2[0], class2), BBA(self.bbs3[0], class3),
                        BBA(self.bbs4[0], class4), BBA(self.bbs5[0], class5), BBA(self.bbs6[0], class6)]

        # Construct final image of the scene by superimposing: bg, img1, img2 and img3
        self.bg = scaleBg.augment_image(bg)
        mask1 = self.img1[:, :, 3]
        self.mask1 = np.stack([mask1] * 3, -1)
        self.final = np.where(self.mask1, self.img1[:, :, 0:3], self.bg)
        mask2 = self.img2[:, :, 3]
        self.mask2 = np.stack([mask2] * 3, -1)
        self.final = np.where(self.mask2, self.img2[:, :, 0:3], self.final)
        mask3 = self.img3[:, :, 3]
        self.mask3 = np.stack([mask3] * 3, -1)
        self.final = np.where(self.mask3, self.img3[:, :, 0:3], self.final)
        mask4 = self.img4[:, :, 3]
        self.mask4 = np.stack([mask4] * 3, -1)
        self.final = np.where(self.mask4, self.img4[:, :, 0:3], self.final)
        mask5 = self.img5[:, :, 3]
        self.mask5 = np.stack([mask5] * 3, -1)
        self.final = np.where(self.mask5, self.img5[:, :, 0:3], self.final)
        mask6 = self.img6[:, :, 3]
        self.mask6 = np.stack([mask6] * 3, -1)
        self.final = np.where(self.mask6, self.img6[:, :, 0:3], self.final)

    def create9CardsScene(self, bg, img1, class1, hulla1, hullb1, img2, class2, hulla2, hullb2, img3, class3, hulla3,
                          hullb3, img4, class4, hulla4, hullb4, img5, class5, hulla5, hullb5, img6, class6, hulla6,
                          hullb6, img7, class7, hulla7, hullb7, img8, class8, hulla8, hullb8, img9, class9, hulla9,
                          hullb9):

        kpsa1 = hull_to_kps(hulla1, decalX3, decalY3)
        kpsb1 = hull_to_kps(hullb1, decalX3, decalY3)
        kpsa2 = hull_to_kps(hulla2, decalX3, decalY3)
        kpsb2 = hull_to_kps(hullb2, decalX3, decalY3)
        kpsa3 = hull_to_kps(hulla3, decalX3, decalY3)
        kpsb3 = hull_to_kps(hullb3, decalX3, decalY3)
        kpsa4 = hull_to_kps(hulla4, decalX3, decalY3)
        kpsb4 = hull_to_kps(hullb4, decalX3, decalY3)
        kpsa5 = hull_to_kps(hulla5, decalX3, decalY3)
        kpsb5 = hull_to_kps(hullb5, decalX3, decalY3)
        kpsa6 = hull_to_kps(hulla6, decalX3, decalY3)
        kpsb6 = hull_to_kps(hullb6, decalX3, decalY3)
        kpsa7 = hull_to_kps(hulla7, decalX3, decalY3)
        kpsb7 = hull_to_kps(hullb7, decalX3, decalY3)
        kpsa8 = hull_to_kps(hulla8, decalX3, decalY3)
        kpsb8 = hull_to_kps(hullb8, decalX3, decalY3)
        kpsa9 = hull_to_kps(hulla9, decalX3, decalY3)
        kpsb9 = hull_to_kps(hullb9, decalX3, decalY3)

        self.img9 = np.zeros((imgH, imgW, 4), dtype=np.uint8)
        self.img9[decalY3:decalY3 + cardH, decalX3:decalX3 + cardW, :] = img9
        self.img9, self.lkps9, self.bbs9 = augment(self.img9, [cardKP, kpsa9, kpsb9], trans_rot8)
        self.img8 = np.zeros((imgH, imgW, 4), dtype=np.uint8)
        self.img8[decalY3:decalY3 + cardH, decalX3:decalX3 + cardW, :] = img8
        self.img8, self.lkps8, self.bbs8 = augment(self.img8, [cardKP, kpsa8, kpsb8], trans_rot7)
        self.img7 = np.zeros((imgH, imgW, 4), dtype=np.uint8)
        self.img7[decalY3:decalY3 + cardH, decalX3:decalX3 + cardW, :] = img7
        self.img7, self.lkps7, self.bbs7 = augment(self.img7, [cardKP, kpsa7, kpsb7], trans_rot6)
        self.img6 = np.zeros((imgH, imgW, 4), dtype=np.uint8)
        self.img6[decalY3:decalY3 + cardH, decalX3:decalX3 + cardW, :] = img6
        self.img6, self.lkps6, self.bbs6 = augment(self.img6, [cardKP, kpsa6, kpsb6], trans_rot5)
        self.img5 = np.zeros((imgH, imgW, 4), dtype=np.uint8)
        self.img5[decalY3:decalY3 + cardH, decalX3:decalX3 + cardW, :] = img5
        self.img5, self.lkps5, self.bbs5 = augment(self.img5, [cardKP, kpsa5, kpsb5], trans_rot4)
        self.img4 = np.zeros((imgH, imgW, 4), dtype=np.uint8)
        self.img4[decalY3:decalY3 + cardH, decalX3:decalX3 + cardW, :] = img4
        self.img4, self.lkps4, self.bbs4 = augment(self.img4, [cardKP, kpsa4, kpsb4], trans_rot1)
        self.img3 = np.zeros((imgH, imgW, 4), dtype=np.uint8)
        self.img3[decalY3:decalY3 + cardH, decalX3:decalX3 + cardW, :] = img3
        self.img3, self.lkps3, self.bbs3 = augment(self.img3, [cardKP, kpsa3, kpsb3], trans_rot2)
        self.img2 = np.zeros((imgH, imgW, 4), dtype=np.uint8)
        self.img2[decalY3:decalY3 + cardH, decalX3:decalX3 + cardW, :] = img2
        self.img2, self.lkps2, self.bbs2 = augment(self.img2, [cardKP, kpsa2, kpsb2], trans_rot3)
        self.img1 = np.zeros((imgH, imgW, 4), dtype=np.uint8)
        self.img1[decalY3:decalY3 + cardH, decalX3:decalX3 + cardW, :] = img1

        while True:
            det_transform_3cards = transform_3cards.to_deterministic()
            _img9, _lkps9, self.bbs9 = augment(self.img9, self.lkps9, det_transform_3cards, False)
            if _img9 is None: continue
            _img8, _lkps8, self.bbs6 = augment(self.img8, self.lkps8, det_transform_3cards, False)
            if _img8 is None: continue
            _img7, _lkps7, self.bbs6 = augment(self.img7, self.lkps7, det_transform_3cards, False)
            if _img7 is None: continue
            _img6, _lkps6, self.bbs6 = augment(self.img6, self.lkps6, det_transform_3cards, False)
            if _img6 is None: continue
            _img5, _lkps5, self.bbs5 = augment(self.img5, self.lkps5, det_transform_3cards, False)
            if _img5 is None: continue
            _img4, _lkps4, self.bbs4 = augment(self.img4, self.lkps4, det_transform_3cards, False)
            if _img4 is None: continue
            _img3, _lkps3, self.bbs3 = augment(self.img3, self.lkps3, det_transform_3cards, False)
            if _img3 is None: continue
            _img2, _lkps2, self.bbs2 = augment(self.img2, self.lkps2, det_transform_3cards, False)
            if _img2 is None: continue
            _img1, self.lkps1, self.bbs1 = augment(self.img1, [cardKP, kpsa1, kpsb1], det_transform_3cards, False)
            if _img1 is None: continue
            break
        self.img9 = _img9
        self.lkps9 = _lkps9
        self.img8 = _img8
        self.lkps8 = _lkps8
        self.img7 = _img7
        self.lkps7 = _lkps7
        self.img6 = _img6
        self.lkps6 = _lkps6
        self.img5 = _img5
        self.lkps5 = _lkps5
        self.img4 = _img4
        self.lkps4 = _lkps4
        self.img3 = _img3
        self.lkps3 = _lkps3
        self.img2 = _img2
        self.lkps2 = _lkps2
        self.img1 = _img1

        self.class1 = class1
        self.class2 = class2
        self.class3 = class3
        self.class4 = class4
        self.class5 = class5
        self.class6 = class6
        self.class6 = class7
        self.class6 = class8
        self.class6 = class9
        self.listbba = [BBA(self.bbs1[0], class1), BBA(self.bbs2[0], class2), BBA(self.bbs3[0], class3),
                        BBA(self.bbs4[0], class4), BBA(self.bbs5[0], class5), BBA(self.bbs6[0], class6),
                        BBA(self.bbs7[0], class7), BBA(self.bbs8[0], class8), BBA(self.bbs9[0], class9)]

        # Construct final image of the scene by superimposing: bg, img1, img2 and img3
        self.bg = scaleBg.augment_image(bg)
        mask1 = self.img1[:, :, 3]
        self.mask1 = np.stack([mask1] * 3, -1)
        self.final = np.where(self.mask1, self.img1[:, :, 0:3], self.bg)
        mask2 = self.img2[:, :, 3]
        self.mask2 = np.stack([mask2] * 3, -1)
        self.final = np.where(self.mask2, self.img2[:, :, 0:3], self.final)
        mask3 = self.img3[:, :, 3]
        self.mask3 = np.stack([mask3] * 3, -1)
        self.final = np.where(self.mask3, self.img3[:, :, 0:3], self.final)
        mask4 = self.img4[:, :, 3]
        self.mask4 = np.stack([mask4] * 3, -1)
        self.final = np.where(self.mask4, self.img4[:, :, 0:3], self.final)
        mask5 = self.img5[:, :, 3]
        self.mask5 = np.stack([mask5] * 3, -1)
        self.final = np.where(self.mask5, self.img5[:, :, 0:3], self.final)
        mask6 = self.img6[:, :, 3]
        self.mask6 = np.stack([mask6] * 3, -1)
        self.final = np.where(self.mask6, self.img6[:, :, 0:3], self.final)
        mask7 = self.img7[:, :, 3]
        self.mask7 = np.stack([mask7] * 3, -1)
        self.final = np.where(self.mask7, self.img7[:, :, 0:3], self.final)
        mask8 = self.img8[:, :, 3]
        self.mask8 = np.stack([mask8] * 3, -1)
        self.final = np.where(self.mask8, self.img8[:, :, 0:3], self.final)
        mask9 = self.img9[:, :, 3]
        self.mask9 = np.stack([mask9] * 3, -1)
        self.final = np.where(self.mask9, self.img9[:, :, 0:3], self.final)

    # def createFanCardsScene(self, bg, image_info_list):
    #
    #     kpsa1 = hull_to_kps(hulla1, decalX3, decalY3)
    #     kpsb1 = hull_to_kps(hullb1, decalX3, decalY3)
    #
    #
    #
    #     self.img4 = np.zeros((imgH, imgW, 4), dtype=np.uint8)
    #     self.img4[decalY3:decalY3 + cardH, decalX3:decalX3 + cardW, :] = img4
    #     self.img4, self.lkps4, self.bbs4 = augment(self.img4, [cardKP, kpsa4, kpsb4], trans_rot1)
    #     self.img3 = np.zeros((imgH, imgW, 4), dtype=np.uint8)
    #     self.img3[decalY3:decalY3 + cardH, decalX3:decalX3 + cardW, :] = img3
    #     self.img3, self.lkps3, self.bbs3 = augment(self.img3, [cardKP, kpsa3, kpsb3], trans_rot2)
    #     self.img2 = np.zeros((imgH, imgW, 4), dtype=np.uint8)
    #     self.img2[decalY3:decalY3 + cardH, decalX3:decalX3 + cardW, :] = img2
    #     self.img2, self.lkps2, self.bbs2 = augment(self.img2, [cardKP, kpsa2, kpsb2], trans_rot3)
    #     self.img1 = np.zeros((imgH, imgW, 4), dtype=np.uint8)
    #     self.img1[decalY3:decalY3 + cardH, decalX3:decalX3 + cardW, :] = img1
    #
    #     while True:
    #         det_transform_3cards = transform_3cards.to_deterministic()
    #         _img4, _lkps4, self.bbs4 = augment(self.img4, self.lkps4, det_transform_3cards, False)
    #         if _img4 is None: continue
    #         _img3, _lkps3, self.bbs3 = augment(self.img3, self.lkps3, det_transform_3cards, False)
    #         if _img3 is None: continue
    #         _img2, _lkps2, self.bbs2 = augment(self.img2, self.lkps2, det_transform_3cards, False)
    #         if _img2 is None: continue
    #         _img1, self.lkps1, self.bbs1 = augment(self.img1, [cardKP, kpsa1, kpsb1], det_transform_3cards, False)
    #         if _img1 is None: continue
    #         break
    #     self.img4 = _img4
    #     self.lkps4 = _lkps4
    #     self.img3 = _img3
    #     self.lkps3 = _lkps3
    #     self.img2 = _img2
    #     self.lkps2 = _lkps2
    #     self.img1 = _img1
    #
    #     self.class1 = class1
    #     self.class2 = class2
    #     self.class3 = class3
    #     self.class4 = class4
    #     self.listbba = [BBA(self.bbs1[0], class1), BBA(self.bbs2[0], class2), BBA(self.bbs3[0], class3),
    #                     BBA(self.bbs3[1], class3), BBA(self.bbs4[1], class4)]
    #
    #     # Construct final image of the scene by superimposing: bg, img1, img2 and img3
    #     self.bg = scaleBg.augment_image(bg)
    #     mask1 = self.img1[:, :, 3]
    #     self.mask1 = np.stack([mask1] * 3, -1)
    #     self.final = np.where(self.mask1, self.img1[:, :, 0:3], self.bg)
    #     mask2 = self.img2[:, :, 3]
    #     self.mask2 = np.stack([mask2] * 3, -1)
    #     self.final = np.where(self.mask2, self.img2[:, :, 0:3], self.final)
    #     mask3 = self.img3[:, :, 3]
    #     self.mask3 = np.stack([mask3] * 3, -1)
    #     self.final = np.where(self.mask3, self.img3[:, :, 0:3], self.final)
    #     mask4 = self.img4[:, :, 3]
    #     self.mask4 = np.stack([mask4] * 3, -1)
    #     self.final = np.where(self.mask4, self.img4[:, :, 0:3], self.final)

    def display(self):
        fig, ax = plt.subplots(1, figsize=(8, 8))
        ax.imshow(self.final)
        for bb in self.listbba:
            rect = patches.Rectangle((bb.x1, bb.y1), bb.x2 - bb.x1, bb.y2 - bb.y1, linewidth=1, edgecolor='b',
                                     facecolor='none')
            ax.add_patch(rect)
        plt.show()

    def res(self):
        return self.final

    def write_files(self, save_dir, card_names, display=False):
        # jpg_fn, xml_fn = give_me_filename(save_dir, ["jpg", "xml"])
        jpg_fn = f"{save_dir}/{card_names}.jpg"
        xml_fn = f"{save_dir}/{card_names}.xml"
        plt.imsave(jpg_fn, self.final)
        if display: print("New image saved in", jpg_fn)
        create_voc_xml(xml_fn, jpg_fn, self.listbba, display=display)
