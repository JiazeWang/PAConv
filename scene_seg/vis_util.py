import os
import sys
import numpy as np
import matplotlib.pyplot as pyplot

colors = {'ceiling':[0,255,0],
          'floor':[0,0,255],
          'wall':[0,255,255],
          'beam':[255,255,0],
          'column':[255,0,255],
          'window':[100,100,255],
          'door':[200,200,100],
          'table':[170,120,200],
          'chair':[255,0,0],
          'sofa':[200,100,100],
          'bookcase':[10,200,100],
          'board':[200,200,200],
          'clutter':[50,50,50]}
colors = list(colors.values())


def write_ply_color(points, labels, out_filename, num_classes=None):
    """ Color (N,3) points with labels (N) within range 0 ~ num_classes-1 as OBJ file """
    labels = labels.astype(int)
    N = points.shape[0]
    if num_classes is None:
        num_classes = np.max(labels) + 1
    else:
        assert (num_classes > np.max(labels))
    fout = open(out_filename, 'w')
    # colors = [pyplot.cm.hsv(i/float(num_classes)) for i in range(num_classes)]
    # colors = [pyplot.cm.jet(i / float(num_classes)) for i in range(num_classes)]
    for i in range(N):
        #c = colors[labels[i]]
        #c = [int(x * 255) for x in c]
        c = colors[labels[i]]
        fout.write('v %f %f %f %d %d %d\n' % (points[i, 0], points[i, 1], points[i, 2], c[0], c[1], c[2]))
    fout.close()


def write_ply_rgb(points, rgb, out_filename, num_classes=None):
    """ Color (N,3) points with labels (N) within range 0 ~ num_classes-1 as OBJ file """
    N = points.shape[0]
    fout = open(out_filename, 'w')
    # colors = [pyplot.cm.hsv(i/float(num_classes)) for i in range(num_classes)]
    # colors = [pyplot.cm.jet(i / float(num_classes)) for i in range(num_classes)]
    for i in range(N):
        #c = colors[labels[i]]
        #c = [int(x * 255) for x in c]
        c = rgb[i]
        fout.write('v %f %f %f %d %d %d\n' % (points[i, 0], points[i, 1], points[i, 2], c[0], c[1], c[2]))
    fout.close()