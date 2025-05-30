import pandas as pd


# remove titles that have duplicated Jones ID
def remove_jones_duplicates(df_titles, df_person_title):
    # make mask with all duplicated Jones
    mask = df_titles['Jones'].duplicated(keep=False)
    # set columns to keep
    columns_to_keep = ['ID_title', 'Jones', 'title', 'translation_of_title']

    # make df with all duplicates
    df_titles_duplicated = df_titles.loc[mask, columns_to_keep]
    # sort df, only Jones with the lowest ID_title will be kept
    df_titles_duplicated = df_titles_duplicated.sort_values(by=["Jones", 'ID_title'])
    df_titles_duplicated.fillna("missing", inplace=True)

    # ID_title with missing or unknown Jones should not be dropped
    omit_jones_list = ['nn', 'missing', 'nn?']
    df_titles_duplicated = df_titles_duplicated.loc[~df_titles_duplicated['Jones'].isin(omit_jones_list), :]

    # make df with ID_title that will be dropped
    df_drop = df_titles_duplicated.loc[df_titles_duplicated['Jones'].duplicated(keep='first'), :]
    df_drop.set_index('ID_title', inplace=True)
    ids_to_drop = df_drop.index.to_list()

    # make df with ID_tile that will be kept
    df_keep = df_titles_duplicated.drop_duplicates(subset='Jones', keep='first')
    df_keep.set_index('Jones', inplace=True)

    # make map where KEY=dropped_ID_title, VALUE:kept_ID_title
    # dtok... dropped to kept
    dtok = {id_title: df_keep.loc[df_drop.loc[id_title, 'Jones'], 'ID_title'] for id_title in ids_to_drop}

    # DROP DUPLICATED TITLES IN df_titles
    df_titles_new = df_titles.loc[~df_titles['ID_title'].isin(ids_to_drop), :]
    # RENAME DROPPED TITLES IN df_person_title
    df_person_title_new = df_person_title.copy()
    df_person_title_new['ID_title'] = df_person_title['ID_title'].map(lambda x: dtok.get(x, x))

    return df_titles_new, df_person_title_new

