import heapq


def show_heatmap(scores):
    pass


def show_instances(scores):
    pass


def show_top_5(plotter, scores, masks):
    top_5 = heapq.nlargest(5, scores, key=scores.get)
    plotter.add_points()
