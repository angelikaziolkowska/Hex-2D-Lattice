from __future__ import print_function
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from scipy import optimize
from CGAL.CGAL_Kernel import Point_2
from CGAL.CGAL_Voronoi_diagram_2 import Voronoi_diagram_2

W = 600
DP, SQ = 2, 2


def _space(k):
    return int(k * 0.4)


def _center(k):
    return _space(k) * 0.25


def getGraphData(u, v, k):
    center, space = _center(k), _space(k)
    actual, dis = {}, {}
    G = nx.triangular_lattice_graph(space, space, periodic=False,
                                    with_positions=True,
                                    create_using=None)
    pos = nx.get_node_attributes(G, 'pos')

    # node locations and distances from 0,0
    for n in G.nodes:
        x = n[0] - center - v[0]
        x2 = n[0] - center
        x = round(float(np.dot(x if (n[1] % 2 == 0) else x2, u[0])), DP)
        y = round(float(np.dot(n[1] - center, v[1])), DP)
        actual[n] = (x, y)
        dis[n] = np.sqrt(np.power(actual[n][0], SQ) + np.power(actual[n][1], SQ))
    dis = dict(sorted(dis.items(), key=lambda item: item[1]))
    dis_items = list(dis.items())
    firstK = dis_items[:k]
    var = {d: G.remove_node(d[0]) for d in dis_items[k:]}
    firstK = [round(column[1], 2) for column in firstK]

    return G, pos, actual, firstK


def _piecewiseLinear(x, x0, x1, b, k1, k2, k3):
    conditions = [x < x0, (x >= x0) & (x < x1), x >= x1]
    functions = [lambda x: k1 * x + b, lambda x: k1 * x + b + k2 * (x - x0),
                 lambda x: k1 * x + b + k2 * (x - x0) + k3 * (x - x1)]
    return np.piecewise(x, conditions, functions)


def graphPiecewiseLinear(k, firstK):
    x = np.arange(1, k + 1, 1, dtype=float)
    y = firstK
    p, e = optimize.curve_fit(_piecewiseLinear, x, y)
    xd = np.linspace(0, k, 100)
    plt.figure(2)
    plt.plot(x, y, "o")
    plt.plot(xd, _piecewiseLinear(xd, *p))
    title = 'Piecewise Linear Graph'
    plt.title(title)
    plt.xlabel('K closest nodes')
    plt.ylabel('Distance from (0,0)')
    plt.savefig(title + '.png')
    img = cv.imread(title + '.png')
    cv.namedWindow(title, cv.WINDOW_AUTOSIZE)
    cv.imshow(title, img)


def graphWithNetworkX(G, pos, actual):
    nx.draw(G, pos=pos, with_labels=False, node_size=40, node_color="#3cff33", width=2, edge_color="red")
    nx.draw_networkx_labels(G, pos=pos,
                            labels={node: (actual[node][0], actual[node][1]) for node in G},
                            font_size=7,
                            font_family="sans-serif",
                            verticalalignment="bottom",
                            horizontalalignment="left")
    plt.figure(1)
    plt.axis('scaled')
    title = "2D Lattice using NetworkX"
    plt.savefig(title + '.png')
    img = cv.imread(title + '.png')
    cv.namedWindow(title, cv.WINDOW_AUTOSIZE)
    cv.imshow(title, img)


def _drawNode(img, center, voronoi=False):
    cv.circle(img,
              center,
              6,
              (60, 255, 51) if not voronoi else (255, 165, 0),
              thickness=-1,
              lineType=8)


def _drawEdge(img, start, end):
    cv.line(img,
            start,
            end,
            (255, 0, 0),
            thickness=2,
            lineType=8)


def graphWithOpenCV(G, pos, k, extraPt):
    title = "2D Lattice and a Single Voronoi Domain using OpenCV"
    vorPts = []
    img = np.ones((int(W * 0.65), int(W * 0.85), 3), dtype=np.uint8) * 255
    j = W // (np.ceil(np.sqrt(k)).astype(int) * 1.6)
    center = _center(k)
    x = (pos[(center, center)][0] + extraPt.x())
    y = (pos[(center, center)][1] + extraPt.y())
    pts = getVoronoiDomainVertices(G, pos, Point_2(float(x), float(y)))

    for e in G.edges:
        start = (int(j * pos[e[0]][0]), int(j * pos[e[0]][1]))
        end = (int(j * pos[e[1]][0]), int(j * pos[e[1]][1]))
        _drawEdge(img, start, end)

    for n in G.nodes:
        _drawNode(img, (int(j * pos[n][0]), int(j * pos[n][1])))

    _drawNode(img, (int(j * x), int(j * y)), True)
    var = {vorPts.append((int(j * p.x()), int(j * p.y()))) for p in pts}
    cv.polylines(img, [np.array(vorPts)], True, (255, 165, 0), 2)

    img = img[::-1, :, :]
    cv.imwrite(title + '.png', img)
    cv.imshow(title, img)


def _getNextVoronoiVertex(e, is_src):
    if is_src and e.has_source():
        return e.source().point()
    elif e.has_target():
        return e.target().point()


def getVoronoiDomainVertices(G, pos, extraLoc):
    points, vert = [], []

    for n in G.nodes:
        points.append(Point_2(pos[n][0], pos[n][1]))
    points.append(extraLoc)
    vd = Voronoi_diagram_2(points)
    assert (vd.is_valid())

    lr = vd.locate(extraLoc)
    if lr.is_face_handle():
        ec_start = lr.get_face_handle().outer_ccb()
        if ec_start.hasNext():
            done = ec_start.next()
            while 1:
                i = ec_start.next()
                v = _getNextVoronoiVertex(i, False)
                if v is not False:
                    vert.append(v)
                if i == done:
                    break
    return np.array(vert)


if __name__ == '__main__':
    u = np.array([1, 0])
    v = np.array([0.5, (np.sqrt(3)*0.5)])
    k = 50
    extraPt = Point_2(0.4, 0.3)

    G, pos, actual, firstK = getGraphData(u, v, k)
    graphWithNetworkX(G, pos, actual)
    graphWithOpenCV(G, pos, k, extraPt)
    graphPiecewiseLinear(k, firstK)

    cv.waitKey(0)
    cv.destroyAllWindows()
