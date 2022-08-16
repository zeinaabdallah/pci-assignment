import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt

directory = r'D:\Zeina\Uni\Bachelor\Thesis\Traces_TimeXY_30sec_txt\Traces_TimeXY_30sec_txt\KAIST'
files = Path(directory).glob('*')
df = [pd.read_csv(file,
                  sep="\t", index_col=False, usecols=[0, 1, 2],
                  names=["Time", "X-coordinate", "Y-coordinate"]) for file in files]
dftot = pd.concat(df)
print(len(dftot.loc[(-1000 <= dftot['X-coordinate']) & (dftot['X-coordinate'] <= 500)]))


def project(dist_exp):
    import pandas as pd
    from pathlib import Path
    from matplotlib import pyplot as plt
    directory = r'D:\Zeina\Uni\Bachelor\Thesis\Traces_TimeXY_30sec_txt\Traces_TimeXY_30sec_txt\KAIST'
    files = Path(directory).glob('*')
    df = [pd.read_csv(file,
                      sep="\t", index_col=False, usecols=[0, 1, 2],
                      names=["Time", "X-coordinate", "Y-coordinate"]) for file in files]
    dftot = pd.concat(df)
    dftot = dftot.loc[(-1000 <= dftot['X-coordinate']) & (dftot['X-coordinate'] <= 500)]
    dftot = dftot.loc[(-300 <= dftot['Y-coordinate']) & (dftot['Y-coordinate'] <= 1200)]
    plt.scatter(dftot['X-coordinate'], dftot['Y-coordinate'], c='b', s=2)
    plt.show()
    print(dftot)
    # number of rows in df = df.shape[0]

    gmm = GaussianMixture(n_components = 3)
    gmm.fit(dftot[['X-coordinate', 'Y-coordinate']])
    clusters = gmm.predict(dftot[['X-coordinate', 'Y-coordinate']])
    dftot['cluster'] = clusters

    plt.scatter(x=dftot["X-coordinate"], y=dftot["Y-coordinate"],s=1, c=dftot["cluster"])
    plt.show()

    from sklearn.cluster import KMeans
    from yellowbrick.cluster import KElbowVisualizer
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(1,10), timings=False)
    visualizer.fit(dftot[['X-coordinate', 'Y-coordinate']])
    visualizer.show()

    km = KMeans(n_clusters=3)
    y_km = km.fit_predict(dftot[['X-coordinate', 'Y-coordinate']])
    dftot['cluster'] = y_km

    print(dftot)

    df1 = dftot[dftot['cluster'] == 0]
    mean1 = df1[['X-coordinate', 'Y-coordinate']].mean()
    df2 = dftot[dftot['cluster'] == 1]
    mean2 = df2[['X-coordinate', 'Y-coordinate']].mean()
    df3 = dftot[dftot['cluster'] == 2]
    mean3 = df3[['X-coordinate', 'Y-coordinate']].mean()

    plt.scatter(df1['X-coordinate'], df1['Y-coordinate'], s=30, c='orange', marker='x', label='cluster 1')
    plt.scatter(df2['X-coordinate'], df2['Y-coordinate'], s=30, c='b', marker='x', label='cluster 2')
    plt.scatter(df3['X-coordinate'], df3['Y-coordinate'], s=30, c='green', marker='x', label='cluster 3')

    plt.scatter(mean1['X-coordinate'], mean1['Y-coordinate'], c='k', marker='+')
    plt.scatter(mean2['X-coordinate'], mean2['Y-coordinate'], c='k', marker='+')
    plt.scatter(mean3['X-coordinate'], mean3['Y-coordinate'], c='k', marker='+')
    plt.show()


    #
    # # # IMP: eps=0.156, min_samples=5
    # # # IMP: eps=0.25, min_samples=6
    # #

    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    coords = dftot[['X-coordinate', 'Y-coordinate']]
    ss = StandardScaler()
    coords = ss.fit_transform(coords)
    db = DBSCAN(eps=0.21, min_samples=6)
    db.fit(coords)
    y_pred = db.fit_predict(coords)
    plt.scatter(coords[:, 0], coords[:, 1], c=y_pred, cmap='Paired')
    plt.title("DBSCAN")
    plt.show()


    from sklearn.metrics import silhouette_score
    score_km = silhouette_score(dftot[['X-coordinate', 'Y-coordinate']], y_km)
    print("KMeans Score is: ", score_km)
    score_DBSCAN = silhouette_score(dftot[['X-coordinate', 'Y-coordinate']], y_pred)
    print("DBSCAN Score is: ", score_DBSCAN)


    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.spatial import Voronoi, voronoi_plot_2d

    def ppp_voronoi(lambda0):
        numPoints = np.random.poisson(lambda0)
        x = np.random.uniform(0, 1500, numPoints)
        y = np.random.uniform(0, 1500, numPoints)
        plt.scatter(x, y, s=5)
        xy = np.stack((x, y), axis=1)
        voronoiData = Voronoi(xy)
        voronoi_plot_2d(voronoiData, show_points=False, show_vertices=False)
        return [x,y,xy,voronoi_plot_2d]

    xx, yy, xxyy, voronoi_plot = ppp_voronoi(100)
    plt.scatter(xx, yy, edgecolor='b', facecolor='b', s=2)
    plt.show()

    from scipy.spatial import Delaunay
    from collections import defaultdict
    import itertools

    tri = Delaunay(xxyy)
    neiList = defaultdict(set)
    for p in tri.vertices:
        for i, j in itertools.combinations(p, 2):
            neiList[i].add(j)
            neiList[j].add(i)
    for key in sorted(neiList.keys()):
        print("%d:%s" % (key, ','.join([str(i) for i in neiList[key]])))

    import networkx as nx

    network = nx.Graph()
    network.add_nodes_from(neiList.keys())
    for i in sorted(neiList.keys()):
        for j in neiList[i]:
            network.add_edge(i, j)
            for k in neiList[j]:
                if i != k:
                    network.add_edge(i, k)
    nx.draw(network, with_labels=True)
    plt.savefig("Graph_Construction.png")


    print(f"density: {nx.density(network)}")

    x_dict = dict(sorted(dict(network.degree).items(), key=lambda item: item[1], reverse=True))
    sorted_vertices = list(x_dict.keys())
    print("Sorted dictonary: ", x_dict)
    print("Sorted vertices: ", sorted_vertices)



    ######### Step 2: Available colors are ordered in a list C

    from random import randint

    colors = []
    for i in range(168):
        colors.append('#%06X' % randint(0, 0xFFFFFF))
    print("List of colors: ", colors)

    ######### Step 3,4: MDFC ALGORITHM
    colored_vertices = []
    MDFC_alg = {}


    def neighbor_with_same_color(MDFC_dict, vertex2, color):
        neighbours = list(network.neighbors(vertex2))
        if len(MDFC_dict) != 0 and len(neighbours) != 0:
            for n in neighbours:
                if n in MDFC_dict.keys():
                    if MDFC_alg[n] == color:
                        return True
        return False


    i = 0
    for vertex1 in sorted_vertices:
        if not (vertex1 in colored_vertices):
            MDFC_alg[vertex1] = colors[i]
            network.nodes[vertex1]['color'] = colors[i]
            colored_vertices.append(vertex1)
            for vertex2 in sorted_vertices:
                neighbour_list = list(network.neighbors(vertex2))
                if (not vertex2 in colored_vertices) and (not vertex1 in neighbour_list) \
                        and not (neighbor_with_same_color(MDFC_alg, vertex2, colors[i])):
                    MDFC_alg[vertex2] = colors[i]
                    network.nodes[vertex2]['color'] = colors[i]
                    colored_vertices.append(vertex2)
                neighbour_list = []
            i += 1

    ncolors_MDFC = set(MDFC_alg.values())
    print("Number of used colors in MDFC: ", len(ncolors_MDFC))

    MDFC_alg = dict(sorted(MDFC_alg.items(), key=lambda item: item[0]))
    print("Each vertex and its color: ", MDFC_alg)

    color_map = []
    for node in network:
        color_map.append(network.nodes[node]['color'])

    nx.draw(network, node_color=color_map, with_labels=True)
    plt.savefig("Graph_Coloring.png")


    A = (nx.to_numpy_matrix(network, nodelist=network.nodes)).astype(int)
    print(f"Adjacency matrix: {A}")

    C = np.full_like(A, 0)
    print(f"Initial forbidden matrix: {C}")

    checker = 1 << network.number_of_nodes() - 1
    checker = bin(checker)[2:]
    checker = np.array(list(checker)).astype(int)

    for i in range(network.number_of_nodes()):
        for j in range(network.number_of_nodes()):
            cek = (np.bitwise_and(C[j], checker))
            if np.all((cek == 0)):
                C[j] = (np.bitwise_or(C[j], A[i]))
                checker = np.roll(checker, 1)
                break

    print(f"Final forbidden matrix: {C}")
    ncolors_bitwise = np.where(C.any(axis=1))[0]
    print("Number of colors in BITWISE: ", len(ncolors_bitwise))

    return [len(ncolors_MDFC), len(ncolors_bitwise), dist_exp, nx.density(network)]
    return [dist_exp, nx.density(network)]


l = list()
for i in range(100, 600, 100):
    l.append(project(i))

l = np.array(l)
print(l)
x = l[:, 0]
y = l[:, 1]


print(x)
print(y)
plt.close()
plt.figure()
plt.plot(x, y)
plt.title('Line Chart of Graph Densities of Different Distribution Expectations')
plt.xlabel('Distribution Expectation (Average Number of Cells)')
plt.ylabel('Density of Graph')
plt.show()

project(100)