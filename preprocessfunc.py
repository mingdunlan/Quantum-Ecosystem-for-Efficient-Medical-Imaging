from skimage.filters import threshold_otsu, gaussian
from numpy import asarray
from skimage.transform import resize
from skimage.segmentation import slic
import streamlit as st
from skimage.color import label2rgb
import matplotlib
import matplotlib.pyplot as plt
# importing gaussian filter and otsu threshold
from skimage.filters import threshold_otsu, gaussian
from skimage.transform import resize
from numpy import asarray
from skimage import color
import pandas as pd

import heartpy as hp
from io import BytesIO
import matplotlib.pyplot as plt

from utils import crop, remove_verticals, remove_bg

sample_rate = 250



def scaling(binary_global):
    min = binary_global.min()
    max = binary_global.max()
    threshold = min + 0.01*(max-min)
    for i in range(0, 300):
        for j in range(0, 450):
            if binary_global[i][j] < threshold:
                binary_global[i][j] = 0
            else:
                binary_global[i][j] = 1


def gen_csv(binary_global):

    pixel_from_top = []
    pixel_from_top.append(160)
    # plt.imshow(binary_global,cmap="gray")
    # print(binary_global)
    # print(binary_global.min())
    # print(binary_global.max())
    for i in range(0, 450):
        id = 0

        for j in range(0, 300):
            if binary_global[j][i] == 0:
                pixel_from_top.append(j)
                id = 1

                break

        if id == 0:
            pixel_from_top.append(pixel_from_top[i-1])

    pixel_from_bottom = []
    for i in range(0, 451):
        pixel_from_bottom.append(160-pixel_from_top[i])

    # plt.xticks(range(0,700,50))
# for i in range (0,480,140):
      # plt.axvline(x=i)
    # plt.plot(pixel_from_bottom)

    arr = asarray(pixel_from_bottom)
    return arr

Lead_params = ['Beats Per Minute(BPM)', 'Interbeat Interval(IBM)','SDNN','SDSD','RMSSD','Proportion of Intervals Between R-R Intervals','MAD','Poincare Analysis','Poincare Plotting']

def analyze(data, ind):
    # run analysis
    wd, m = hp.process(data, sample_rate)

    plot_object = hp.plotter(wd, m, figsize=(
        20, 10), show=False, title="Lead {} Peak Detection and Analysis".format(ind))

    buf = BytesIO()
    plot_object.savefig(buf, format="jpg")
    st.image(buf, width=800, channels="BGR")

    measures = list(m.values())
    
    i = 0

    for i in range(len(Lead_params)):

        if Lead_params[i]=='SDNN' or  Lead_params[i]=='SDSD' or Lead_params[i]=='RMSSD' or Lead_params[i]=='MAD' or (Lead_params[i]=='Beats Per Minute(BPM)' and measures[i]>220):
            continue
        else:
            st.subheader('%s:' % (Lead_params[i]))
            st.write(' %f' % (measures[i]))
        i = i+1


def lead_func(image):
    Lead_1 = image[300:600, 150:643]
    Lead_2 = image[300:600, 646:1135]
    Lead_3 = image[300:600, 1140:1625]
    Lead_4 = image[300:600, 1630:2125]
    Lead_5 = image[600:900, 150:643]
    Lead_6 = image[600:900, 646:1135]
    Lead_7 = image[600:900, 1140:1625]
    Lead_8 = image[600:900, 1630:2125]
    Lead_9 = image[900:1200, 150:643]
    Lead_10 = image[900:1200, 646:1135]
    Lead_11 = image[900:1200, 1140:1625]
    Lead_12 = image[900:1200, 1630:2125]
    Lead_13 = image[1250:1480, 150:2125]

    Leads = [Lead_1, Lead_2, Lead_3, Lead_4, Lead_5, Lead_6,
             Lead_7, Lead_8, Lead_9, Lead_10, Lead_11, Lead_12, Lead_13]

    Lead_info = ['Lead Observation of lateral aspect of left ventricle',
              'Observation of inferior aspect of left ventricle',
              'Observation of inferior aspect of left ventricle',
              'Observation of right atrium and cavity of left ventricle',
              'Observation of lateral aspect of left ventricle',
              'Observation of inferior aspect of left ventricle',
              'Observation of ventricular septum',
              'Observation of ventricular septum',
              'Observation of anterior wall of left ventricle',
              'Observation of anterior wall of left ventricle',
              'Observation of lateral wall of left ventricle',
              'Observation of lateral wall of left ventricle',
              ]

    # plotting lead 1-12
    fig, ax = plt.subplots(4, 3)

    fig.set_size_inches(15, 15)

    x_counter = 0
    y_counter = 0

    for x, y in enumerate(Leads[:len(Leads)-1]):
        if (x+1) % 3 == 0:
            ax[x_counter][y_counter].imshow(y)
            ax[x_counter][y_counter].axis('off')
            ax[x_counter][y_counter].set_title("Lead {}".format(x+1))
            x_counter += 1
            y_counter = 0
        else:
            ax[x_counter][y_counter].imshow(y)
            ax[x_counter][y_counter].axis('off')
            ax[x_counter][y_counter].set_title("Lead {}".format(x+1))
            y_counter += 1

    # plot the image

    # importing gaussian filter and otsu threshold
    list_bins = []
    # creating subplot of size(4,3) 4 rows and 3 columns
    fig2, ax2 = plt.subplots(4, 3)

    fig2.set_size_inches(20, 20)

    # setting counter for plotting based on value
    x_counter = 0
    y_counter = 0

    # looping through image list containg all leads from 1-12
    for x, y in enumerate(Leads[:len(Leads)-1]):
        # converting to gray scale
        grayscale = color.rgb2gray(y)
        # smoothing image
        blurred_image = gaussian(grayscale, sigma=0.7)
        # thresholding to distinguish foreground and background
        # using otsu thresholding for getting threshold value
        global_thresh = threshold_otsu(blurred_image)

        # creating binary image based on threshold
        binary_global = 1 - (blurred_image < global_thresh)
        # resize image
        binary_global = resize(binary_global, (300, 450))
        crop(binary_global)
        remove_verticals(binary_global)
        list_bins.append(binary_global)

    ind = 0

    for binary_global in list_bins:
        
        
        scaling(binary_global)
        arr = gen_csv(binary_global)
        pd.DataFrame(arr).to_csv('./'+str(ind)+'.csv', index=False)

        data = hp.get_data('./'+str(ind)+'.csv')

        ind = ind+1

        st.subheader('Lead {}'.format(ind))

        st.write(Lead_info[ind])

        analyze(data, ind)
