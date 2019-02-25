"""
@author: Theophile BORAUD
t.boraud@warwick.co.uk
Copyright 2019, Theophile BORAUD, Anthony STROCK, All rights reserved.
"""

import random
import numpy as np
import math
from PIL import Image
import freetype as ft
import scipy.ndimage
import sys



# ------------------------------------------- #
# ------------------ CLASS ------------------ #
# ------------------------------------------- #


class Alphascii:


# ------------------------------------------- #
# ---------------- FUNCTIONS ---------------- #
# ------------------------------------------- #


    def __init__(self, mode, size, seed = None, set_i = 0):
        """
        Constructor function

        Args:
            mode (string): Mode for the sequence generation ("Training" or "Testing")
            size (int): Number of characters in the sequence

        Returns:
            Alphascii object
        """

        self.seed = self.set_seed(seed)
        self.alphabet = "abcdefghijklmnopqrstuvwxyz0123456789 !\"#$%&\'()*+,.-_/:;<=>?@€[]|§"

        #self.fontfiles = ["data/font/FreeMono.ttf", "data/font/FreeMono_Italic.ttf", "data/font/FreeMono_Bold.ttf", "data/font/FreeMono_BoldItalic.ttf"]
        self.fontfiles = ["data/font/FreeMono_BoldItalic.ttf"]
        self.fontstring = ""
        for i in range(len(self.fontfiles)):
            str = self.fontfiles[i][19:-4]
            if str == "":
                str = "Classic"
            str += "_"
            self.fontstring += str
        self.mode = mode
        self.fontstring = self.fontstring[:-1]
        self.nested_lvl_max = 6 # Curly bracket maximum nested level
        self.nested_lvl = np.empty(size, dtype = np.int64) # WM-units
        self.width_chars = np.empty(size, dtype = np.int64) # Output units
        self.width_total = 0 # Total length in pixels of the sequence image
        self.sequence = self.random_sequence(mode, size, set_i)
        self.n_brackets = self.count_characters("Brackets") # Number of curly brackets into the sequence
        self.n_characters = self.count_characters("Characters") # Number of characters other than curly brackets into the sequence
        self.data, self.image = self.convert_sequence_to_img(self.sequence, zmin = 0.9, zmax=1.1)
        self.sequence_pxl = self.get_sequence_pxl() # Characters for each pixel column in the sequence
        self.nested_lvl_outputs = self.compute_nested_lvl_outputs() #
        self.sequence_outputs = self.compute_sequence_outputs()
        self.n_input = self.data.shape[1]

    def set_seed(self, seed = None):
        """
        Create the seed (for random values) variable if None

        Args:
            seed (int): Numpy seed if given, else generate random one

        Returns:
            int: Seed
        """

        if seed is None:
            import time
            seed = int((time.time()*10**6) % 4294967295)
            print("Random seed for alphascii:", seed)
        try:
            np.random.seed(seed)
        except:
            print("!!! WARNING !!!: Alphascii seed was not set correctly.")
        return seed


    def random_sequence(self, mode, size, set_i):
        """
        Generate a random sequence using the Alphascii alphabet

        Args:
            mode (string): "Training" for a training sequence, "Testing" for testing
            size (int): Number of characters composing the sequence

        Returns:
            string: Sequence generated
        """

        sequence = "" # Store the sequence of characters
        i = 0 # Nested level of curly brackets
        j = random.randint(0, 64) # Current character index
        if mode == "Training":
            probs = [70, 85]
        elif mode == "Testing":
            probs = [94, 97]
        elif mode == "PCA":
            probs = [101, 101] # No curly brackets
            i = set_i

        for k in range(size):
            char = None
            while(char is None): # Select a randon probability until the next character has been randomly chosen
                prob_char = random.randint(1, 100) # Percent probability of next character

                if prob_char <= probs[0]: # Chance of getting an ASCII character
                    prob_ascii = random.randint(1, 10) # Percent probability to predict or randomise the next ASCII character
                    next_j = (i + j + 1) % len(self.alphabet)
                    if prob_ascii <= 8: # 80% chance next character index is i + j modulo 65
                        j = next_j
                    else: # 20% chance next character is randomly selected among the 64 other alphabet characters
                        rand_index = next_j
                        while rand_index == next_j:
                            rand_index = random.randint(0, len(self.alphabet) - 1)
                        j = rand_index
                    char = self.alphabet[j]
                # Chance of getting an open curly bracket
                elif probs[0] < prob_char <= probs[1] and i < self.nested_lvl_max:
                    char = '{'
                    i += 1 # Increase the nexted level

                # Chance of getting a close curly bracket
                elif probs[1] < prob_char and i > 0:
                    char = '}'
                    i -= 1 # Decrease the nested level

            self.nested_lvl[k] = i
            sequence += char

        return sequence


    def count_characters(self, mode):
        """
        Count the number of curly brackets or other characters in the sequence

        Args:
            mode (string): "Brackets" to count curly brackets, "Characters" for others

        Return:
            int: Number of target characters to count in the sequence
        """

        counter = 0
        if mode == "Brackets":
            for i in range(len(self.sequence)):
                if self.sequence[i] == "{" or self.sequence[i] == "}":
                    counter += 1

        elif mode == "Characters":
            for i in range(len(self.sequence)):
                if self.sequence[i] != "{" and self.sequence[i] != "}":
                    counter += 1

        return counter


    def convert_sequence_to_img(self, text, size=11, zmin=1.0, zmax=1.0, add_kerning=False):
        """
        Generate a noisy bitmap string of text using different fonts

        Args:
            text (string): Text to be displayed
            size (int): Font size to use to generate text (default 20)
            zmin (float): Minimal horizontal distortion
            zmax (float): Maximal horizontal distortion

        Returns:
            Tuple of numpy array (Z,I)
                Z is the bitmap string array
                I is a unidimensional numpy array that indicates the corresponding
           character for each column of Z
        """

        # Load fonts
        fonts = self.fontfiles
        faces = [ft.Face(filename) for filename in fonts]
        for face in faces:
            face.set_char_size(size*64)
        slots = [face.glyph for face in faces]

        # Find baseline and height (maximum)
        baseline, height = 0, 0
        for face in faces:
            ascender = face.size.ascender >> 6
            descender = face.size.descender >> 6
            height = max(height, ascender-descender)
            baseline = max(baseline, -descender)


        # Set individual character font and zoom level
        font_index = np.random.randint(0, len(faces), len(text))
        zoom_level = np.random.uniform(zmin, zmax, len(text))

        # First pass to compute bounding box
        width = 0
        previous = 0
        for i,c in enumerate(text):
            index = font_index[i]
            zoom = zoom_level[i]
            face, slot = faces[index], slots[index]
            face.load_char(c, ft.FT_LOAD_RENDER | ft.FT_LOAD_FORCE_AUTOHINT)

            bitmap = slot.bitmap
            kerning = face.get_kerning(previous, c).x >> 6
            kerning = int(round(zoom*kerning))
            advance = slot.advance.x >> 6
            advance = int(round(zoom*advance))

            if i == len(text)-1:
                width += max(advance, int(round(zoom*bitmap.width)))
            else:
                width += advance + kerning
            previous = c



        # Allocate arrays for storing data
        Z = np.zeros((height,width), dtype=np.ubyte)
        I = np.zeros(width, dtype=np.int) + ord(' ')

        # Second pass for actual rendering
        x, y = 0, 0
        previous = 0
        for i,c in enumerate(text):
            index = font_index[i]
            zoom = zoom_level[i]
            face, slot = faces[index], slots[index]
            face.load_char(c, ft.FT_LOAD_RENDER | ft.FT_LOAD_FORCE_AUTOHINT)

            bitmap = slot.bitmap
            top, left = slot.bitmap_top, slot.bitmap_left
            w,h = bitmap.width, bitmap.rows
            y = height - baseline - top
            kerning = 0
            if(add_kerning):
                kerning = face.get_kerning(previous, c).x >> 6
                kerning = int(round(zoom*kerning))

            advance = slot.advance.x >> 6
            advance = int(round(zoom*advance))

            glyph = np.array(bitmap.buffer, dtype='ubyte').reshape(h,w)
            glyph = scipy.ndimage.zoom(glyph, (1, zoom), order=3)
            #print(np.max(np.float64(glyph) * 255 / np.max(glyph)))
            #print(glyph.shape)
            if glyph.shape[0]!= 0 and glyph.shape[1]!= 0:
                glyph = np.uint8(np.float64(glyph) * 255.0 / np.max(glyph))
            w = glyph.shape[1]

            x += kerning
            left = 0
            Z[y:y+h,x+left:x+left+w] += glyph
            I[x:x+w] = ord(c)
            x += advance
            previous = c
            self.width_chars[i] = advance
            self.width_total += advance

        # TODO: Salt and pepper
        Z = self.salt_pepper(Z[:,:self.width_total])
        img = Image.fromarray(Z, 'L')
        #print(np.asarray(img).shape)
        #img = img.resize((w_font, 12))
        #print(np.asarray(img).shape)

        return (Z/255.0).T, img


    def salt_pepper(self, data):
        """
        Add salt and pepper noise to the given data

        Inputs:
            data (matrix): Given data to add noise to

        Returns:
            matrix: salt and pepper noised data

        """
        p = 0.5
        x = 0.1
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                rng = np.random.uniform(low = 0, high = 1) <= p
                rng2 = np.random.choice([-1, 1])
                data[i, j] = int(np.clip(np.float(data[i,j]) + x * rng * rng2 * 255, 0, 255))

        return data


    def compute_nested_lvl_outputs(self):
        """
        Compute the matrix containing the current bracket nested lvl for each column of pixel

        Returns:
            width_total x nested_lvl_max: Brackets levels (1 for opened, 0 for closed) during time (number of width pixels) of the sequence
        """

        nested_lvl_outputs = np.ones((self.width_total, self.nested_lvl_max))
        nested_lvl_outputs *= -0.5
        current_pixel = 0

        current_bracket_lvl = 0
        for i in range(len(self.nested_lvl)):
            for j in range(self.width_chars[i]):
                if j >= np.ceil(self.width_chars[i]/2):
                    current_bracket_lvl = self.nested_lvl[i]
                for k in range(current_bracket_lvl):
                    nested_lvl_outputs[current_pixel, k] = 0.5
                current_pixel += 1

        return nested_lvl_outputs


    def compute_sequence_outputs(self):
        """
        Compute the matrix containing the current character activation for each column of pixel

        Returns:
            width_total x nested_lvl_max: Brackets levels (1 for opened, 0 for closed) during time (number of width pixels) of the sequence
        """

        sequence_outputs = np.nan * np.zeros((self.width_total, len(self.alphabet)))
        current_pixel = -1

        for i in range(len(self.sequence)):
            if i != (len(self.sequence) - 1):
                #current_char = sequence[i]
                current_char = self.sequence[i + 1]
            else:
                current_char = "{"
            mid_char = math.ceil(self.width_chars[i]/2)
            current_pixel += mid_char


            for j in range(len(self.alphabet)):
                if current_char == self.alphabet[j]:
                    sequence_outputs[current_pixel] = 0
                    #sequence_outputs[current_pixel, (j + 1 + nested_lvl[i]) % len(self.alphabet)] = 1
                    sequence_outputs[current_pixel, j] = 1
            current_pixel += (self.width_chars[i] - mid_char)

        return sequence_outputs


    def get_sequence_pxl(self):
        """
        Compute the characters for each pixel column in the sequence data

        Returns:
            width_total sized array: Return each characters represented by a given pixel column
        """

        sequence_pxl = np.empty(self.width_total, dtype="U1")
        current_char = 0
        current_width = 0
        for i in range(len(sequence_pxl)):
            sequence_pxl[i] = self.sequence[current_char]
            current_width += 1
            if current_width == self.width_chars[current_char]:
                current_width = 0
                current_char += 1

        return sequence_pxl


# ------------------------------------------- #
# ----------------- TESTING ----------------- #
# ------------------------------------------- #

if __name__ == "__main__":
    alphascii = Alphascii("PCA", 100, set_i = 0)
    print(alphascii.sequence)
    alphascii.image.show()
