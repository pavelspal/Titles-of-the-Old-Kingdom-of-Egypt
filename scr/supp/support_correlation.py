
# make a heatmap
def heat_map(corr_matrix, save_path='img/', name=''):
    # corr_matrix ... data
    # save_path ... path where to store outputs plots
    # name ... name of the output file
    import os                                   # making dir
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle    # making rectangles into heatmap
    import seaborn as sns                       # heatmap

    # plotting function
    def make_plot(data, case, save_path=save_path):
        # data ... data to be plotted
        # case ... total counts or probability plot?
        # save_path ... path where to store outputs plots
        global ax       # axis
        # make dataframe
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data, columns=labels, index=labels)
        n, m = data.shape

        make_annot = (n < 30)               # plot annotation within cells?
        count_column = (n != m)             # do df contain column with 'total count'?
        folder = ''

        # FONT SIZE, OTHER SETTINGS
        title_size = 20
        subtitle_size = 16
        label_size = 16
        ticks_size = 16
        count_size = 12
        plt.figure(figsize=(15, 12))

        # PLOT HEATMAP
        # make annotation with total counts
        if make_annot and count_column:
            folder = 'normalized/'

            fake_data = data.copy()
            # color 'total count' same as diagonal elements
            fake_data.iloc[:, -1] = 1.0

            label = data[data.columns[:-1]].copy()
            label = label.map(lambda x: f'{x:.2f}')
            label = pd.concat([label, data[data.columns[-1]]], axis=1)

            ax = sns.heatmap(data=fake_data,
                             annot=label,
                             fmt='',
                             annot_kws={"fontsize": count_size})
            # Customizing annotation text alignment for the last column 'total count'
            for i, text in enumerate(ax.texts):
                if i % m == m - 1:  # Check if it's the last column
                    text.set_ha('right')  # Align text to the right
                    text.set_x(text.get_position()[0] + 0.4)  # Adjust x position to move text more to the right
        # make annotation without total counts
        elif make_annot:
            folder = 'counts_small/'
            ax = sns.heatmap(data=data,
                             annot=True,
                             fmt='d' if np.all(np.equal(data, data.astype(int))) else '.2f',
                             annot_kws={"fontsize": count_size})
        # no annotation at all
        else:
            folder = 'counts_big/'
            ax = sns.heatmap(data=data,
                             annot=False)

        # HIGHLIGHT DIAGONAL CELLS AND COLUMN 'total count'
        if make_annot:
            for i in range(n):
                ax.add_patch(Rectangle((i, i), 1, 1, fill=False, edgecolor='white', lw=2))
                if count_column:
                    ax.add_patch(Rectangle((m - 1, i), 1, 1, fill=False, edgecolor='white', lw=2))

        # SET LABELS
        plt.title(f'Correlation of {name}', fontsize=title_size)
        plt.suptitle(f'{case}', fontsize=subtitle_size)
        plt.xlabel('Title', fontsize=label_size)
        plt.ylabel('Title', fontsize=label_size)
        plt.yticks(rotation=0)
        # Adjust tick labels font size
        plt.xticks(fontsize=ticks_size)  # Adjust the fontsize as needed
        plt.yticks(fontsize=ticks_size)  # Adjust the fontsize as needed
        # Set font size of color bar labels
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=ticks_size)  # Adjust the fontsize as needed
        # Adjusting the layout to make the plot more tight
        plt.tight_layout()

        # SAVE AS PNG AND PDF
        save_path = save_path + folder
        if len(save_path) > 0 and not os.path.exists(save_path):
            # If not, create the folder
            os.makedirs(save_path[:-1])
        plt.savefig(save_path + f'Correlation - {name} - {case}.png')
        plt.savefig(save_path + f'Correlation - {name} - {case}.pdf')
        plt.show()

    # prepare data
    labels = corr_matrix.columns
    labels = [label[:30] for label in labels]   # shorten labels
    corr = corr_matrix.to_numpy()

    # total counts
    make_plot(corr, case='Total counts')
    # probability
    norm = corr.diagonal().reshape((-1, 1))                # norm == diagonal elements
    row_norm = pd.DataFrame(corr/norm, columns=labels, index=labels)
    row_norm['total count'] = norm.flatten()                # save norm into dataframe
    make_plot(row_norm, case='Row-normalized counts')


# split titles according to the first dash '-' in the name
def split_titles(titles):
    import re
    import pandas as pd

    # handle too long title name
    exception_1 = 'epithet referring to the restricted access in cultic context'
    exception_1 = exception_1.lower()

    exception_epithet = [
        'epithet connected with employment'
        'epithet connected with afterworld',
        'epithet connected with afterworld',
        'epithet connected with employment',
        'epithet connected with royal ceremonies',
        'epithet connected with the family',
        'epithet connected with the god',
        "epithet expressing king's favour"
        ]

    p_subjob = re.compile(pattern=r"""
            (?P<category>\b[\w ]*?\b)    # General job name
            (?:\s* [–-] \s*)         # Whitespace, minus (dash), whitespace 
            (?P<sub_category>\b.*\b)      # More detailed job name
            """, flags=re.VERBOSE)
    p_only_dash = re.compile(pattern=r"""
            (?P<category>\b[\w ]*?\b)    # General job name
            (?:\s* [–] \s*)         # Whitespace, minus (dash), whitespace 
            """, flags=re.VERBOSE)

    titles_dict = []
    for title in titles:
        title = title.lower()
        # exception for too long name, occurs only once
        if title == exception_1:
            titles_dict.append({'category': 'epithet',
                                'sub_category': 'epithet ... cultic context'})
            continue
        if title in exception_epithet:
            titles_dict.append({'category': 'epithet',
                                'sub_category': 'epithet ... exception_epithet'})
            continue

        se_subjob = p_subjob.search(title)
        se_dash = p_only_dash.search(title)
        if se_subjob:
            # If matched, extract the parts and append to the result list
            titles_dict.append(se_subjob.groupdict())
        elif se_dash:
            # If only dash matched, append the string and 'None' to the result list
            titles_dict.append({'category': se_dash.groupdict()['category'],
                                'sub_category': 'no subcategory'})
        else:
            # If not matched, append the string and 'None' to the result list
            titles_dict.append({'category': title,
                                'sub_category': 'no subcategory'})
    return pd.DataFrame(titles_dict)


# describe pivot table
def describe(pivot_table):
    import numpy as np
    import pandas as pd
    des = pd.DataFrame(pivot_table.to_numpy().flatten()).describe()
    print(des)
    # 'max' must be 1 otherwise there are duplicates
    if des.iloc[-1, 0] != 1:
        print("\033[92mERROR in max occurrence. There might be duplicates!\033[0m")

