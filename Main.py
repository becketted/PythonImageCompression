# imports
import os
from PIL import Image
from PIL import ImageChops
from scipy.fftpack import dct
from scipy.fftpack import idct
import math, operator
import functools
import numpy as np


# Mapping and Un-Mapping
# Contains use of the Quantization and De-Quantization methods
def Mapper(RGB):

    # get size of image
    size = RGB[0].size
    # check that image is divisible by 8 for blocks
    if size[0] % 8 != 0 & size[1] % 8 != 0:
        # if not then display error & exit
        print("Image Wrong Size")
        return 0
    else:
        # set block size
        nSize = 8

    # iterate through each channel (R, G, B)
    for channel in RGB:
        # access pixels
        pixels = channel.load()

        # loop through width
        for i in range(0, size[0], nSize):
            # loop through height
            for j in range(0, size[1], nSize):

                # create empty neighbourhood
                neighbourhood = [[0] * nSize for x in range(nSize)]

                # fill neighbourhood with pixel data
                for k in range(nSize):
                    for l in range(nSize):
                        neighbourhood[k][l] = pixels[i+k,j+l]

                # compute discrete cosine transformation coefficients for the neighbourhood across both axis.
                neighbourhood = dct(dct(neighbourhood, axis=0, norm="ortho").tolist(), axis=1, norm="ortho").tolist()

                # quantize dct coefficients
                neighbourhood = Quantizer(neighbourhood)

                # place neighbourhood back in pixels
                for k in range(nSize):
                    for l in range(nSize):
                        pixels[i+k,j+l] = neighbourhood[k][l]
    return RGB

def UnMap(RGB):

    # get image size and set block size
    size = RGB[0].size
    nSize = 8

    # iterate through each channel
    for channel in RGB:
        # load pixels
        pixels = channel.load()

        # loop through width
        for i in range(0, size[0], nSize):
            # loop through height
            for j in range(0, size[1], nSize):

                # create neighbourhood
                neighbourhood = [[0] * nSize for x in range(nSize)]

                # fill neighbourhood
                for k in range(nSize):
                    for l in range(nSize):
                        neighbourhood[k][l] = pixels[i+k,j+l]

                # dequantize
                neighbourhood = DeQuantize(neighbourhood)

                # inverse discrete cosine transformation (across both axis)
                neighbourhood = idct(idct(neighbourhood, axis=0, norm="ortho"), axis=1, norm="ortho").tolist()

                # round and cap values
                for k in range(nSize):
                    for l in range(nSize):
                        neighbourhood[k][l] = round(neighbourhood[k][l])
                        # if values are above 255 (out of range) then cap them.
                        if neighbourhood[k][l] > 255:
                            neighbourhood[k][l] = 255

                # put neighbourhood into back into image
                for k in range(nSize):
                    for l in range(nSize):
                        pixels[i+k,j+l] = neighbourhood[k][l]
    return RGB


# Quantization and De-quantization
def Quantizer(neighbourhood):

    # create an empty neighbourhood
    qtNeighbourhood = [[0] * 8 for x in range(8)]

    # The quantization matrix controls compression ration & quality.
    # The below matrix is a 50% quality reduction as specified in the JPEG standard.
    QM = [[16,11,10,16,24,40,51,61],
        [12,12,14,19,26,58,60,55],
        [14,13,16,24,40,57,69,56],
        [14,17,22,29,51,87,80,62],
        [18,22,37,56,68,109,103,77],
        [24,35,55,64,81,104,113,92],
        [49,64,78,87,103,121,120,101],
        [7,92,95,98,112,100,103,99]]

    # another matrix for testing
    QM2 = [[17,18,24,47,99,99,99,99],
        [18,21,26,66,99,99,99,99],
        [24,26,56,99,99,99,99,99],
        [47,66,99,99,99,99,99,99],
        [99,99,99,99,99,99,99,99],
        [99,99,99,99,99,99,99,99],
        [99,99,99,99,99,99,99,99],
        [99,99,99,99,99,99,99,99]]

    # iterate through the neighbourhood
    for k in range(8):
        for l in range(8):
            # divide coefficient by appropriate value in the matrix, then round the result
            qtNeighbourhood[k][l] = round(neighbourhood[k][l] / QM[k][l])
            #qtNeighbourhood[k][l] = round(neighbourhood[k][l] / QM2[k][l])
    # return the neighbourhood
    return qtNeighbourhood

def DeQuantize(neighbourhood):

    # create an empty neighbourhood
    qtNeighbourhood = [[0] * 8 for x in range(8)]

    # The quantization matrix controls compression ration & quality.
    # The below matrix is a 50% quality reduction as specified in the JPEG standard.
    QM  = [[16,11,10,16,24,40,51,61],
            [12,12,14,19,26,58,60,55],
            [14,13,16,24,40,57,69,56],
            [14,17,22,29,51,87,80,62],
            [18,22,37,56,68,109,103,77],
            [24,35,55,64,81,104,113,92],
            [49,64,78,87,103,121,120,101],
            [7,92,95,98,112,100,103,99]]

    # another matrix for testing
    QM2 = [[17, 18, 24, 47, 99, 99, 99, 99],
           [18, 21, 26, 66, 99, 99, 99, 99],
           [24, 26, 56, 99, 99, 99, 99, 99],
           [47, 66, 99, 99, 99, 99, 99, 99],
           [99, 99, 99, 99, 99, 99, 99, 99],
           [99, 99, 99, 99, 99, 99, 99, 99],
           [99, 99, 99, 99, 99, 99, 99, 99],
           [99, 99, 99, 99, 99, 99, 99, 99]]

    # iterate through the neighbourhood
    for k in range(8):
        for l in range(8):
            # reverse the quantisation by inversing it (multiplying the coefficient by the matrix value)
            qtNeighbourhood[k][l] = neighbourhood[k][l] * QM[k][l]
            #qtNeighbourhood[k][l] = neighbourhood[k][l] * QM2[k][l]

    # return the neighbourhood
    return qtNeighbourhood


# Encoding and Decoding (Run Length)
def RunLengthEncode(RGB):
    # create empty container
    reRGB = []
    # for each image channel (R, G, B)
    for channel in RGB:
        # obtain access to the pixels of the layer
        pixels = channel.load()
        # get the size of the image
        size = channel.size
        # create an empty string for pixel data
        linearstring = ''

        # loop through height
        for j in range(size[1]):
            # loop through width
            for i in range(size[0]):
                # add the pixel value to the string
                linearstring += str(pixels[i,j]) + ' '

        # create an empty string for encoded data
        encoded = ''
        # split the pixel data by spaces
        data = linearstring.split()
        # initialise count
        count = 0
        # for each pixel
        for i in data:
            if i == "0":
                # if number is zero, increase count of zeros
                count += 1
            else:
                # if number isn't zero
                if count != 0:
                    # if zero count is greater than 0, implying previous zeros
                    if count == 1:
                        # if count only equals 1 (only one zero)
                        # then just add one zero to the string
                        encoded += "0" + " "
                    else:
                        # if count is greater than 1, print out zero then the number of zeros.
                        encoded += "0" + str(count) + " "
                # if the number isn't zero and there is no previous count then print next number
                encoded += i + " "
                # reset count
                count = 0

        # repeat for the end of each layer as it gets missed otherwise
        # if last number isn't zero
        if count != 0:
            # if zero count is greater than 1
            if count == 1:
                # if count only equals 1 (only one zero)
                # then just add one zero to the string
                encoded += "0" + " "
            else:
                # if count is greater than 1, print out zero then count.
                encoded += "0" + str(count) + " "
        # add encoded string to the rgb object as a layer
        reRGB.append(encoded)
    return reRGB

def RunLengthDecode(ppm):
    # retrieve pixel data
    data = ppm[4].split()
    # create empty containers
    RGB = []
    pixels = []
    # for each element in the pixel data
    for i in data:
        # convert the number to a string so that we can process it
        number = str(i)
        # check the first character for zeros
        if number[0] == "0":
            # if first character is zero
            if int(number) == 0:
                # if the whole number is zero then add zero to the pixel list
                pixels.append(number[0])
            else:
                # if the whole number is not zero
                for i in range(int(number)):
                    # for the size of the number eg. 50, add zero to the pixel list n times
                    pixels.append(number[0])
        else:
            # if the number doesn't begin with zero, add it to the pixel list
            pixels.append(int(number))

    # get image dimensions
    width = int(ppm[1])
    height = int(ppm[2])
    layersize = width * height

    # for each layer of the image
    for layer in range(3):
        # create a new single channel image with the given dimensions
        image = Image.new(mode="L", size=(width, height))
        # insert the slice of decoded data corresponding to the layer
        image.putdata(pixels[layer*layersize:(layer+1)*layersize-1])
        # add the layer
        RGB.append(image)

    return RGB


# Splitting & Merging Channels
def splitRGB(ppm):
    # if the ppm is an rgb image
    if ppm.mode == "RGB":
        # split channels
        R,G,B = ppm.split()
        # place in list for returning
        RGB = [R, G, B]
        return RGB

def mergeRGB(RGB):
    # merge channels back together
    ppm = Image.merge("RGB", (RGB[0], RGB[1], RGB[2]))
    return ppm


# I/O Functions
def writePPM(fileName, ppm):
    # check if .ppm given in file name
    if ".ppm" in fileName:
        # write file
        ppm.save(fileName)
    else:
        # add .ppm if missing and write
        ppm.save(fileName, ppm)

def writeCompressed(fileName, ppm, RGB):
    # write compressed output
    f = open(fileName, "w")
    # don't necessarily need the ppm encoding
    f.write("P6\n")
    # write the image dimensions
    f.write(str(ppm.size[0]) + " " + str(ppm.size[1]) + "\n")
    # don't necessarily need the ppm max value
    f.write("255\n")
    # for each channel in the image (R, G, B)
    for channel in RGB:
        # write out the run length encoded contents
        f.write(channel)
    f.close()

def readPPM(fileName):

    # check if file is a ppm
    if ".ppm" in fileName:
        # open image
        ppm = Image.open(fileName)
    else:
        # print error and quit program
        print("Invalid Filetype")
        return 0
    return ppm

def readCompressed(fileName):
    ppm = []
    f = open(fileName, "r")
    # don't necessarily need the ppm standard information...
    encoding = f.readline()
    ppm.append(encoding)

    # get image size (w x h)
    imagesize = f.readline().split()
    ppm.append(imagesize[0])
    ppm.append(imagesize[1])

    # ppm max value - again, probably not needed.
    maxval = f.readline()
    ppm.append(maxval)

    # the pixel data
    imagedata = f.read()
    ppm.append(imagedata)
    return ppm


def RMSE(img1, img2):
    # from: http://effbot.org/zone/pil-comparing-images.htm
    # calculate the root-mean-square difference between two images
    h = ImageChops.difference(img1, img2).histogram()

    # calculate rms (round to 5dp for ease of reading)
    return round(math.sqrt(functools.reduce(operator.add,map(lambda h, i: h * (i ** 2), h, range(256))) / (float(img1.size[0]) * img1.size[1])),5)

def CompressionRatio(filepath1, filepath2):
    # get number of bytes for original image
    sizeBefore = os.path.getsize(filepath1)
    # get number of bytes for compressed output
    sizeAfter = os.path.getsize(filepath2)
    # calculate compression ratio
    ratio = sizeBefore / sizeAfter
    # return ratio rounded to 4dp
    return round(ratio,4)

def SignalToNoise(a, axis=0, ddof=0):
    # used to be in scipy but was depreciated
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m / sd)


# Test function
def Test():
    # read image
    ppm = readPPM("2.ppm")
    img1 = ppm

    # split into rgb
    RGB = splitRGB(ppm)

    # remove spatial redundancy, to some extent psychovisual redundancy too
    RGB = Mapper(RGB)

    # encode using run length encoding, removing coding redundancy
    RGB = RunLengthEncode(RGB)

    # write compressed output to file
    writeCompressed("2export", ppm, RGB)

    # read in the compressed file
    ppm = readCompressed("2export")

    # reverse the run length encoding
    RGB = RunLengthDecode(ppm)

    # reverse the mapping and quantisation
    RGB = UnMap(RGB)

    # reassemble image
    ppm = mergeRGB(RGB)

    # write decompressed output to file
    writePPM("2export.ppm", ppm)


    # Compression stats
    img2 = readPPM("2export.ppm")
    print("Details of Compressed Output:")
    print("Compression Ratio: " + str(CompressionRatio("2.ppm","2export")) + ":1")
    print("Size before compression (bytes): " + str(os.path.getsize("2.ppm")))
    print("Size after compression (bytes): " + str(os.path.getsize("2export")))
    print("\nFurther statistics: ")
    print("Root Mean Squared Error between Images: " + str(RMSE(img1, img2)))
    print("Signal to Noise Ratio of Original image: " + str(SignalToNoise(img1, axis=None)))
    print("Signal to Noise Ratio of Compressed image: " + str(SignalToNoise(img2, axis=None)))
    print("Difference in Signal to Noise Ratio: " + str(SignalToNoise(img2, axis=None) - SignalToNoise(img1, axis=None)))


# Main
def main():
    Test()

# Code to make the main function run properly
if __name__== "__main__":
    main()
