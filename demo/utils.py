import heapq


def show_heatmap(scores):
    pass


def show_instances(scores):
    pass


def clean(dict1, dict2):
    keys_to_remove = [key for key in dict1 if len(dict1[key]) == 0]

    for key in keys_to_remove:
        del dict1[key]
        del dict2[key]


def show_top_20_instances(plotter, point_cloud, scores, masks):
    plotter.set_color_cycler(
        ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink', 'brown', 'orange', 'teal',
         'coral', 'lightblue', 'lime', 'lavender', 'turquoise', 'darkgreen', 'tan', 'salmon', 'gold', 'lightpurple',
         'darkred', 'darkblue'])
    clean(masks, scores)
    top_20 = heapq.nlargest(20, scores, key=scores.get)

    for i in top_20:
        plotter.add_points(point_cloud.points[masks[i], :], render_points_as_spheres=True,
                           point_size=3)
