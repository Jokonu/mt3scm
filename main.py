from plot_examples import plot_curvature_torsion_example, plot_lorenz_attractor_3d, publication_plot_random_examples, publication_plot_agglomerative_examples
from plot_perfect import publication_plot_metric_perfect, publication_plot_metric_agglom_example
from plot_results_analysis import analyse_correlation

GRAPHICS_FORMAT = "png"  # pdf, png, eps
RESOLUTION_DPI = 800

def main():
    # koehn2.pdf
    plot_lorenz_attractor_3d(plot_name="koehn2", graphics_format=GRAPHICS_FORMAT, resolution_dpi=RESOLUTION_DPI)
    # koehn3.pdf fig:lorenz-attractor-original
    plot_curvature_torsion_example(plot_name="koehn3", graphics_format=GRAPHICS_FORMAT, resolution_dpi=RESOLUTION_DPI)
    # koehn4.pdf
    publication_plot_metric_perfect(plot_name="koehn4", graphics_format=GRAPHICS_FORMAT, resolution_dpi=RESOLUTION_DPI)
    # koehn5.pdf
    publication_plot_metric_agglom_example(plot_name="koehn5", graphics_format=GRAPHICS_FORMAT, resolution_dpi=RESOLUTION_DPI)
    # koehn6.pdf, needs 'python -m plot_examples --kmeans --agglomerative --random' to run first
    analyse_correlation(plot_name="koehn6", graphics_format=GRAPHICS_FORMAT, resolution_dpi=RESOLUTION_DPI)
    # koehn7.pdf fig:metric-random. Need to copy the results into table when newly plottet, because of randomness
    publication_plot_random_examples(plot_name="koehn7", graphics_format=GRAPHICS_FORMAT, resolution_dpi=RESOLUTION_DPI)
    # koehn8.pdf
    publication_plot_agglomerative_examples(plot_name="koehn8", graphics_format=GRAPHICS_FORMAT, resolution_dpi=RESOLUTION_DPI)


if __name__ == "__main__":
    # plot_lorenz_attractor_3d(plot_name="koehn2", graphics_format=GRAPHICS_FORMAT, resolution_dpi=RESOLUTION_DPI)
    # plot_curvature_torsion_example(plot_name="koehn3", graphics_format=GRAPHICS_FORMAT, resolution_dpi=RESOLUTION_DPI)
    # publication_plot_metric_perfect(plot_name="koehn4", graphics_format=GRAPHICS_FORMAT, resolution_dpi=RESOLUTION_DPI)
    # publication_plot_metric_agglom_example(plot_name="koehn5", graphics_format=GRAPHICS_FORMAT, resolution_dpi=RESOLUTION_DPI)
    # analyse_correlation(plot_name="koehn6", graphics_format=GRAPHICS_FORMAT, resolution_dpi=RESOLUTION_DPI)
    # publication_plot_random_examples(plot_name="koehn7", graphics_format=GRAPHICS_FORMAT, resolution_dpi=RESOLUTION_DPI)
    # publication_plot_agglomerative_examples(plot_name="koehn8", graphics_format=GRAPHICS_FORMAT, resolution_dpi=RESOLUTION_DPI)
    main()