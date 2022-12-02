import matplotlib.pyplot as plt


def plot_observable_list_same_axis(x, y_list, xlabel, ylabel, legend, figname, title=None, xticks_list=None, fontsize=14,
                                   vertical_lines=None, legend_ncols=2, make_average=False, figsize=(22.86, 9.43),
                                   markersize=4.5, bbox_to_anchor=(0.5, -0.15), tight_layout=True, show=True):
    plt.style.use('seaborn-whitegrid')
    f, ax = plt.subplots(figsize=figsize)

    for y in y_list:
        ax.plot(x, y, marker='o', markersize=markersize)

    ax.tick_params(axis='x', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)

    if vertical_lines:
        for line_dict in vertical_lines:
            x = line_dict['x']
            label = line_dict['label']
            plt.axvline(x=x, label=label, linestyle='--', linewidth=3)
            plt.text(x, 0, label, fontsize=fontsize)

    if title:
        ax.set_title(title, fontsize=fontsize)

    if xticks_list:
        ax.set_xticks(xticks_list, fontsize=fontsize)

    if make_average:
        legend_ncols += 1
        legend = legend + ['Average']
        ax.plot(x, np.nanmean(y_list, axis=0).tolist())

    ax.legend(legend, loc="lower center", bbox_to_anchor=bbox_to_anchor, ncol=legend_ncols, fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    f.subplots_adjust(bottom=0.25)
    
    if tight_layout:
        plt.tight_layout()
    
    plt.savefig(figname)

    if show:
        plt.show()
    else:
        plt.close()
