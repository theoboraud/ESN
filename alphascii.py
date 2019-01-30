import random
import numpy as np
import math
from PIL import Image, ImageDraw, ImageFont
import freetype as ft
import scipy.ndimage



# ------------------------------------------- #
# ------------------ CLASS ------------------ #
# ------------------------------------------- #


class Alphascii:



# ------------------------------------------- #
# ---------------- FUNCTIONS ---------------- #
# ------------------------------------------- #


    def __init__(self, mode, size, seed = None):
        """
        Constructor function

        Args:
            mode (string): Mode for the sequence generation ("Training" or "Testing")
            size (int): Number of characters in the sequence

        Returns:
            Alphascii object
        """
        self.seed = self.set_seed(seed)
        self.alphabet = "abcdefghijklmnopqrstuvwxyz0123456789 !\"#$%&\'()*+,.-_/:;<=>?@€|§"

        #self.fontfiles = ["data/font/FreeMono.ttf", "data/font/FreeMono_Italic.ttf", "data/font/FreeMono_Bold.ttf", "data/font/FreeMono_Bold_Italic.ttf"]
        self.fontfiles = ["data/font/FreeMono.ttf", "data/font/FreeMono_Italic.ttf"]
        self.nested_lvl_max = 6 # Curly bracket maximum nested level
        self.nested_lvl = np.empty(size, dtype = np.int64) # WM-units
        self.width_chars = np.empty(size, dtype = np.int64) # Output units
        self.width_total = 0 # Total length in pixels of the sequence image
        self.sequence = self.random_sequence(mode, size)
        self.n_brackets = self.count(self.sequence, "Brackets") # Number of curly brackets into the sequence
        self.n_characters = self.count(self.sequence, "Characters") # Number of characters other than curly brackets into the sequence
        self.data, self.image = self.convert_sequence_to_img(self.sequence, size = 11, zmin = 0.9, zmax=1.1)
        self.sequence_time = self.get_sequence_time(self.sequence, self.width_chars)
        self.nested_lvl_pxl = self.compute_nested_lvl_pxl(self.nested_lvl, self.width_chars, self.width_total)
        self.sequence_pxl = self.compute_sequence_pxl(self.sequence, self.width_chars, self.width_total, self.nested_lvl)
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
        try:
            np.random.seed(seed)
        except:
            print("!!! WARNING !!!: Alphascii seed was not set correctly.")
        return seed


    def random_sequence(self, mode, size):
        """
        Generate a random sequence using the Alphascii alphabet

        Args:
            size (int): Number of characters composing the sequence

        Returns:
            string: Sequence generated
        """
        if mode == "Training":
            probs = [70, 85]
        elif mode == "Testing":
            probs = [94, 97]

        sequence = "" # Store the sequence of characters
        i = 0 # Nested level of curly brackets
        j = random.randint(0, 64) # Current character index
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


    def count(self, sequence, mode):

        counter = 0
        for i in range(len(sequence)):
            if mode == "Brackets":
                if sequence[i] == "{" or sequence[i] == "}":
                    counter += 1
            elif mode == "Characters":
                if sequence[i] != "{" and sequence[i] != "}":
                    counter += 1
        return counter


    #def convert_char_to_img(self, c, w_font):
        """
        Convert a given char into an image with a randomly selected font

        Args:
            c (char): Character to convert into an image
            width (int): randomly selected font width

        Returns:
            Image object: Character image
        """

        """ TO KICK OUT
        fontpath = "data/font/"
        rand_font = random.randint(0, len(self.fontfiles) - 1)
        if fontfile is None:
            fontfile = fontpath + self.fontfiles[rand_font]
        if fontfile == "data/font/FreeMono_Bold.ttf" or fontfile == "data/font/FreeMono_Bold_Italic.ttf":
            font = ImageFont.truetype(fontfile, 11)
        else:
            font = ImageFont.truetype(fontfile, 100)
        fontsize = font.getsize(c)
        ascent, descent = font.getmetrics()

        img = Image.new('L', (fontsize[0], ascent + descent), 0)

        d = ImageDraw.Draw(img)

        d.text((0, 0), c, fill=255, font = font)
        img = resizeimage.resize_width(img, width)
        #img = img.resize((width, 12))#, Image.ANTIALIAS)

        return img
        """
        """
        size = 20
        faces = ( ft.Face('data/font/FreeMono.ttf'),
                  ft.Face('data/font/FreeMono_Bold.ttf'),
                  ft.Face('data/font/FreeMono_Italic.ttf'),
                  ft.Face('data/font/FreeMono_Bold_Italic.ttf') )
        #faces = [ft.Face('data/font/FreeMono_Bold.ttf'), ft.Face('data/font/FreeMono.ttf'), ft.Face('data/font/FreeMono_Bold_Italic.ttf')]
        for face in faces:
            face.set_char_size( size*64 )
        #faces[0].set_char_size( size*64 )
        slots = [face.glyph for face in faces]
        #slots = [faces[0].glyph]

        text = c
        font = np.random.randint(0, len(faces), len(text))


        # First pass to compute bbox
        width, height, baseline = 0, 0, 0
        previous = 0
        for i,char in enumerate(text):
            index = font[i]
            face, slot = faces[index], slots[index]
            face.load_char(char, ft.FT_LOAD_RENDER | ft.FT_LOAD_FORCE_AUTOHINT)
            bitmap = slot.bitmap
            height = max(height,
                         bitmap.rows + max(0,-(slot.bitmap_top-bitmap.rows)))
            baseline = max(baseline, max(0,-(slot.bitmap_top-bitmap.rows)))
            kerning = face.get_kerning(previous, char)
            width += (slot.advance.x >> 6) #+ (kerning.x >> 6)
            previous = char

        Z = np.zeros((height,width), dtype=np.ubyte)

        # Second pass for actual rendering
        x, y = 0, 0
        previous = 0
        for i,char in enumerate(text):
            index = font[i]
            face, slot = faces[index], slots[index]
            face.load_char(char, ft.FT_LOAD_RENDER | ft.FT_LOAD_FORCE_AUTOHINT)
            bitmap = slot.bitmap
            top = slot.bitmap_top
            left = slot.bitmap_left
            w,h = bitmap.width, bitmap.rows
            y = height-baseline-top
            kerning = face.get_kerning(previous, char)
            x += (kerning.x >> 6)
            Z[y:y+h,x:x+w] += np.array(bitmap.buffer, dtype='ubyte').reshape(h,w)
            x += (slot.advance.x >> 6)
            previous = char

        #Z = Z/255.0
        img = Image.fromarray(Z, 'L')
        #print(np.asarray(img).shape)
        img = img.resize((w_font, 12))
        #print(np.asarray(img).shape)

        return img
        """

    def convert_sequence_to_img(self, text, size=11, zmin=1.0, zmax=1.0, add_kerning=False):
        """
        Generate a noisy bitmap string of text using different fonts

        Parameters
        ==========

        text: string
            Text to be displayed

        size: int
            Font size to use to generate text (default 20)

        zmin: float
            Minimal horizontal distortion

        zmax: float
            Maximal horizontal distortion

        Returns
        =======

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

        Z = Z[:,:self.width_total]
        img = Image.fromarray(Z, 'L')
        #print(np.asarray(img).shape)
        #img = img.resize((w_font, 12))
        #print(np.asarray(img).shape)

        return (Z/255.0).T, img


    #def convert_sequence_to_img(self, s):
        """
        Convert a given string sequence into an image by adding each character image one by one

        Args:
            s (string): Sequence to convert into an image

        Returns:
            Image object: Sequence image
        """
        """
        for i in range(len(s)):
            self.width_chars[i] = 7 #random.randint(6, 8)
            self.width_total += self.width_chars[i]


        img = Image.new('L', (self.width_total, 30), 0)
        width_sequence = 0
        for i in range(len(s)):
            img.paste(self.convert_char_to_img(s[i], w_font=self.width_chars[i]), (width_sequence, 0))
            width_sequence += self.width_chars[i]


        return img
        """

    def compute_nested_lvl_pxl(self, nested_lvl, width_chars, width_total):
        """
        Compute the matrix containing the current bracket nested lvl for each column of pixel

        Args:
            nested_lvl (size vector): Bracket nested level for each character in the sequence
            width_chars (size vector): Width in pixels for each character in the sequence
            width_total (int): Total length of the sequence image in pixel

        Returns:
            width_total x nested_lvl_max: Brackets levels (1 for opened, 0 for closed) during time (number of width pixels) of the sequence
        """
        nested_lvl_pxl = np.ones((width_total, self.nested_lvl_max))
        nested_lvl_pxl *= -0.5
        current_pixel = 0

        current_bracket_lvl = 0
        for i in range(len(nested_lvl)):
            for j in range(width_chars[i]):
                if j >= np.ceil(width_chars[i]/2):
                    current_bracket_lvl = nested_lvl[i]
                for k in range(current_bracket_lvl):
                    nested_lvl_pxl[current_pixel, k] = 0.5
                current_pixel += 1

        return nested_lvl_pxl


    def compute_sequence_pxl(self, sequence, width_chars, width_total, nested_lvl):
        """
        Compute the matrix containing the current character activation for each column of pixel

        Args:
            nested_lvl (size vector): Bracket nested level for each character in the sequence
            width_chars (size vector): Width in pixels for each character in the sequence
            width_total (int): Total length of the sequence image in pixel

        Returns:
            width_total x nested_lvl_max: Brackets levels (1 for opened, 0 for closed) during time (number of width pixels) of the sequence

        """
        sequence_pxl = np.nan * np.zeros((width_total, len(self.alphabet)))
        current_pixel = -1

        for i in range(len(sequence)):
            if i != (len(sequence) - 1):
                #current_char = sequence[i]
                current_char = sequence[i + 1]
            else:
                current_char = "{"
            mid_char = math.ceil(width_chars[i]/2)
            current_pixel += mid_char


            for j in range(len(self.alphabet)):
                if current_char == self.alphabet[j]:
                    sequence_pxl[current_pixel] = 0
                    #sequence_pxl[current_pixel, (j + 1 + nested_lvl[i]) % len(self.alphabet)] = 1
                    sequence_pxl[current_pixel, j] = 1
            current_pixel += (width_chars[i] - mid_char)

        return sequence_pxl



    def get_sequence_time(self, sequence, width_chars):
        sequence_time = np.empty(self.width_total, dtype="U1")
        current_char = 0
        current_width = 0
        for i in range(len(sequence_time)):
            sequence_time[i] = sequence[current_char]
            current_width += 1
            if current_width == width_chars[current_char]:
                current_width = 0
                current_char += 1
        return sequence_time


# ------------------------------------------- #
# ----------------- TESTING ----------------- #
# ------------------------------------------- #
if __name__ == "__main__":
    alphascii = Alphascii("Training", 100)
    print(alphascii.sequence)
    alphascii.image.show()
    print("width_total:", alphascii.width_total)
    print(alphascii.sequence_time)
