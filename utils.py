def remove_verticals(img) :
  x1_start = 503
  x1_end = 506
  y1_start = 125
  y1_end = 825

  img[ y1_start : y1_end, x1_start : x1_end ] = 255

  x2_start = 995
  x2_end = 998
  y2_start = 125
  y2_end = 825

  img[ y2_start : y2_end, x2_start : x2_end ] = 255

  x3_start = 1486
  x3_end = 1490
  y3_start = 125
  y3_end = 825

  img[ y3_start : y3_end, x3_start : x3_end ] = 255


def remove_bg(img) : 
  for i in range(img.shape[0]):
    for j in range(img.shape[1]) :
      if img[i][j] <50 :
        img[i][j] = 0
      else :
        img[i][j] = 255

def crop(img) : 
  x_start = 140
  x_end = -40
  y_start = 287
  y_end = -57

  img = img[ y_start : y_end, x_start : x_end ]
  return img