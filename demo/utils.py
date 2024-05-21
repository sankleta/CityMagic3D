import heapq

COLORS = ['darkred', 'brown', 'red', 'salmon', 'coral', 'orange', 'gold', 'yellow', 'lime', 'green', 'darkgreen',
          'teal', 'turquoise', 'cyan', 'lightblue', 'blue', 'darkblue', 'purple', 'lavender', 'magenta',
          'pink']

SCALE_RAINBOW_COLORS = [
    "#FF0000",  # Red
    "#FF7F00",  # Orange
    "#FFFF00",  # Yellow
    "#7FFF00",  # Light Green
    "#00FF00",  # Green
    "#00FF7F",  # Spring Green
    "#00FFFF",  # Cyan
    "#007FFF",  # Azure
    "#0000FF",  # Blue
    "#7F00FF"  # Violet
]


def show_heatmap(plotter, mesh, scores, masks):
    plotter.add_text("Showing heatmap in rainbow style, from red (the lowest score) to violet (the highest score)",
                     font_size=10, position="lower_left")
    clean(masks, scores)
    for i in masks:
        if scores[i] > 0:
            plotter.add_points(mesh.points[masks[i], :], color=get_color(scores[i]), render_points_as_spheres=True,
                               point_size=3)


def get_color(score):
    # Calculate the index in the color list. Assuming the score range is from 0 to 0.5
    normalized_score = score / 0.5
    index = int(normalized_score * (len(SCALE_RAINBOW_COLORS) - 1))
    return SCALE_RAINBOW_COLORS[index]


# Remove masks with empty projection and their embeddings
def clean(dict1, dict2):
    keys_to_remove = [key for key in dict1 if len(dict1[key]) == 0]

    for key in keys_to_remove:
        del dict1[key]
        del dict2[key]


def show_top_20_instances(plotter, mesh, scores, masks):
    plotter.add_text("Showing top 20 matching instances", font_size=10, position="lower_left")
    plotter.set_color_cycler(COLORS)
    clean(masks, scores)
    top_20 = heapq.nlargest(20, scores, key=scores.get)

    for i in top_20:
        plotter.add_points(mesh.points[masks[i], :], render_points_as_spheres=True, point_size=3)
