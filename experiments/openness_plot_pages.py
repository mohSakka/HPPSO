"""Four-panel pages shared by openness reports of any suite size."""
from pathlib import Path


def make_pages(count, figsize=(15, 10)):
    import matplotlib.pyplot as plt
    pages, axes = [], []
    for first in range(0, count, 4):
        active = min(4, count - first)
        if active == 1:
            figure, axis = plt.subplots(figsize=(figsize[0], figsize[1] * .65), constrained_layout=True)
            flat = [axis]
        elif active == 2:
            figure, grid = plt.subplots(1, 2, figsize=(figsize[0], figsize[1] * .6), constrained_layout=True)
            flat = list(grid.flat)
        else:
            figure, grid = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
            flat = list(grid.flat)
        for unused in flat[active:]:
            unused.set_visible(False)
        pages.append(figure)
        axes.extend(flat[:active])
    return pages, axes


def save_pages(pages, folder, basename, title, pdf=None):
    import matplotlib.pyplot as plt
    folder = Path(folder)
    filenames = []
    for index, figure in enumerate(pages):
        page_title = title + (f"\nPage {index + 1}/{len(pages)}" if len(pages) > 1 else "")
        figure.suptitle(page_title, fontsize=12)
        filename = basename + (f"_{index + 1:02d}" if index else "") + ".png"
        figure.savefig(folder / filename, dpi=170)
        if pdf is not None:
            pdf.savefig(figure)
        plt.close(figure)
        filenames.append(filename)
    return filenames
