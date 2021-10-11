import pandas as pd
import os

pd.set_option("display.max_columns", None)


def read_tweet_class_label(tweet_file, include_meta_data=True):
    class_label_df = pd.read_csv(tweet_file, sep="\t")

    if include_meta_data:
        base = os.path.basename(tweet_file)
        base_parts = str(base).split("_")

        if len(base_parts) == 2:
            base_0 = base_parts[0]
            base_1 = base_parts[1].split(".")[0]

            class_label_df["event_type"] = base_0
            class_label_df["data_type"] = base_1

    return class_label_df


def read_tweet_events(tweet_file, include_meta_data=True):
    tweet_events_df = pd.read_csv(tweet_file, sep="\t")

    if include_meta_data:
        base = os.path.basename(tweet_file)
        base_parts = str(base).split("_")

        if len(base_parts) > 0:
            base_event_name = "_".join(base_parts[0:-1])
            base_data_type = base_parts[-1].split(".")[0]

            tweet_events_df["event_name"] = base_event_name
            tweet_events_df["data_type"] = base_data_type


    return tweet_events_df


def get_files_from_folder(folder):
    files_list = os.listdir(folder)
    all_files = list()

    for entry in files_list:
        full_path = os.path.join(folder, entry)

        if os.path.isdir(full_path):
            all_files = all_files + get_files_from_folder(full_path)
        else:
            all_files.append(full_path)

    return all_files


def read_all_tweet_class_labels(events_folder_file_list, include_meta_data=True):
    all_tweet_class_labels_df = pd.DataFrame()
    first_file = True

    for tweet_file in events_folder_file_list:
        if tweet_file.endswith(".tsv"):
            if first_file:
                all_tweet_class_labels_df = read_tweet_class_label(tweet_file, include_meta_data=include_meta_data)

            else:
                temp_class_label_df = read_tweet_class_label(tweet_file, include_meta_data=include_meta_data)
                all_tweet_class_labels_df = pd.concat([all_tweet_class_labels_df, temp_class_label_df], axis=0)
            first_file = False

    return all_tweet_class_labels_df


def read_all_tweet_data_events_sets(events_set_folder_file_list, include_meta_data=True):
    all_tweet_data_events_sets_df = pd.DataFrame()
    first_file = True

    for tweet_file in events_set_folder_file_list:
        if tweet_file.endswith(".tsv"):
            if first_file:
                all_tweet_data_events_sets_df = read_tweet_events(tweet_file, include_meta_data=include_meta_data)

            else:
                temp_class_label_df = read_tweet_events(tweet_file, include_meta_data=include_meta_data)
                all_tweet_data_events_sets_df = pd.concat([all_tweet_data_events_sets_df, temp_class_label_df], axis=0)
            first_file = False

    return all_tweet_data_events_sets_df


def get_consolidated_disaster_tweet_data(root_directory="data/",
                                         event_type_directory="HumAID_data_event_type",
                                         events_set_directories=["HumAID_data_events_set1_47K",
                                                                 "HumAID_data_events_set2_29K"],
                                         include_meta_data=True):

    events_folder = os.path.join(root_directory, event_type_directory) # "../../../data/HumAID/HumAID_data_event_type.tar/event_type/"
    events_folder_file_list = get_files_from_folder(events_folder)
    all_tweet_class_labels_df = read_all_tweet_class_labels(events_folder_file_list=events_folder_file_list,
                                                            include_meta_data=include_meta_data)

    events_set_folder_file_list = []
    for event_set_directory in events_set_directories:
        full_event_set_directory = os.path.join(root_directory, event_set_directory)
        temp_events_set_folder_file_list = get_files_from_folder(full_event_set_directory)
        events_set_folder_file_list.extend(temp_events_set_folder_file_list)

    all_tweet_data_events_sets_df = \
        read_all_tweet_data_events_sets(events_set_folder_file_list=events_set_folder_file_list,
                                        include_meta_data=include_meta_data)

    consolidated_disaster_tweet_data_df = pd.merge(all_tweet_class_labels_df,
                                                   all_tweet_data_events_sets_df[["tweet_id", "tweet_text"]],
                                                   on="tweet_id", how="left")

    return consolidated_disaster_tweet_data_df



if __name__ == "__main__":
    # # Test read_tweet_class_label(tweet_file) **********************************************************************
    # tweet_file_all_train = "../../../data/HumAID/HumAID_data_all_combined.tar/all_combined/all_train.tsv"
    # earthquake_train_tweet_file = "../../../data/HumAID/HumAID_data_event_type.tar/event_type/earthquake_train.tsv"
    #
    # earthquake_train_df = read_tweet_class_label(tweet_file=earthquake_train_tweet_file, include_meta_data=True)
    # print("earthquake_train_df :")
    # print(earthquake_train_df.head())
    # print()
    #
    # all_train_df = read_tweet_class_label(tweet_file=tweet_file_all_train, include_meta_data=True)
    # print("all_train_df :")
    # print(all_train_df.head())
    # print()
    #
    # print("all_train_df[all_train_df['tweet_id'] == 1176502390863933440] :")
    # print(all_train_df[all_train_df['tweet_id'] == 1176502390863933440])
    #
    # print("earthquake_train_df[earthquake_train_df['tweet_id'] == 1176502390863933440] :")
    # print(earthquake_train_df[earthquake_train_df['tweet_id'] == 1176502390863933440])
    # ****************************************************************************************************************

    # # Test read_tweet_events() **********************************************************************
    # # tweet_file = "../../../data/HumAID/HumAID_data_events_set1_47K.tar/events_set1/canada_wildfires_2016/canada_wildfires_2016_train.tsv"
    # tweet_file = "../../../data/HumAID/HumAID_data_events_set2_29K.tar/events_set2/pakistan_earthquake_2019/pakistan_earthquake_2019_train.tsv"
    # # tweet_file = "../../../data/HumAID/HumAID_data_events_set2_29K.tar/events_set2/pakistan_earthquake_2019/pakistan_earthquake_2019_test.tsv"
    # tweet_file = "../../../data/HumAID/HumAID_data_events_set2_29K.tar/events_set2/pakistan_earthquake_2019/pakistan_earthquake_2019_dev.tsv"
    # tweet_events_df = read_tweet_events(tweet_file=tweet_file, include_meta_data=True)
    # print("tweet_events_df :")
    # print(tweet_events_df.head())
    # print("tweet_events_df.dtypes :")
    # print(tweet_events_df.dtypes)
    # ****************************************************************************************************************

    # Test read_all_tweet_class_labels() *************************************************************************
    # folder = "../../../data/HumAID/HumAID_data_event_type.tar/event_type/"
    # all_files = get_files_from_folder(folder)
    # all_tweet_class_labels_df = read_all_tweet_class_labels(events_folder_file_list=all_files, include_meta_data=True)
    #
    # print("all_tweet_class_labels_df.shape :", all_tweet_class_labels_df.shape)
    # print()
    #
    # print("all_tweet_class_labels_df :")
    # print(all_tweet_class_labels_df.head())
    # print()
    #
    # print("all_tweet_class_labels_df[all_tweet_class_labels_df['tweet_id'] == 1176502390863933440] :")
    # print(all_tweet_class_labels_df[all_tweet_class_labels_df['tweet_id'] == 1176502390863933440])
    # ****************************************************************************************************************

    # Test get_consolidated_disaster_tweet_data() **********************************************************************
    consolidated_disaster_tweet_data_df = \
        get_consolidated_disaster_tweet_data(root_directory="../data/",
                                             event_type_directory="HumAID_data_event_type",
                                             events_set_directories=["HumAID_data_events_set1_47K",
                                                                     "HumAID_data_events_set2_29K"],
                                             include_meta_data=True)

    print("consolidated_disaster_tweet_data_df.shape :", consolidated_disaster_tweet_data_df.shape)
    print()

    print("consolidated_disaster_tweet_data_df :")
    print(consolidated_disaster_tweet_data_df.head())
    print()

    print("consolidated_disaster_tweet_data_df[consolidated_disaster_tweet_data_df['tweet_id'] == 1176502390863933440] :")
    print(consolidated_disaster_tweet_data_df[consolidated_disaster_tweet_data_df['tweet_id'] == 1176502390863933440])
    print()

    print("consolidated_disaster_tweet_data_df.isnull().sum() :")
    print(consolidated_disaster_tweet_data_df.isnull().sum())
    print()

    # consolidated_disaster_tweet_data_df.to_csv("../data/consolidated_disaster_tweet_data.tsv", sep="\t", index=False)
    consolidated_disaster_tweet_data_values = consolidated_disaster_tweet_data_df[["tweet_id", "tweet_text"]].values
    print("consolidated_disaster_tweet_data_values :")
    print(consolidated_disaster_tweet_data_values)

    print("consolidated_disaster_tweet_data_values[0][0] :", consolidated_disaster_tweet_data_values[0][0])
    # ****************************************************************************************************************