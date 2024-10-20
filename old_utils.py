from json import JSONDecodeError
from os import listdir
from os.path import isfile, join
import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from collections import defaultdict
from datetime import datetime
import pandas as pd
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from plot import *


def count_json_level1_entries():
    ds_path = 'ds_sequences_no_empty_activities/train/Benign'
    json_path_list = [join(ds_path, f) for f in listdir(ds_path) if isfile(join(ds_path, f))]
    ds_path = 'ds_sequences_no_empty_activities/test/Benign'
    json_path_list.extend([join(ds_path, f) for f in listdir(ds_path) if isfile(join(ds_path, f))])
    activity = np.array([])
    receiver = np.array([])
    service = np.array([])
    provider = np.array([])
    i = 0
    for json_path in json_path_list:
        with open(json_path) as f:
            json_example = json.load(f)
            activity = np.append(activity, [len(json_example['activity'])])
            receiver = np.append(receiver, [len(json_example['receiver'])])
            service = np.append(service, [len(json_example['service'])])
            provider = np.append(provider, [len(json_example['provider'])])
            f.close()
        i += 1
        if i % 100 == 0:
            print(str(i) + ' examples computed')
    print('\n\n\n')

    print('ACTIVITY')
    print('max: ' + str(np.max(activity)))
    print('min: ' + str(np.min(activity)))
    print('mean: ' + str(np.mean(activity)))
    print('def stand: ' + str(np.std(activity)))

    print('RECEIVER')
    print('max: ' + str(np.max(receiver)))
    print('min: ' + str(np.min(receiver)))
    print('mean: ' + str(np.mean(receiver)))
    print('def stand: ' + str(np.std(receiver)))

    print('SERVICE')
    print('max: ' + str(np.max(service)))
    print('min: ' + str(np.min(service)))
    print('mean: ' + str(np.mean(service)))
    print('def stand: ' + str(np.std(service)))

    print('PROVIDER')
    print('max: ' + str(np.max(provider)))
    print('min: ' + str(np.min(provider)))
    print('mean: ' + str(np.mean(provider)))
    print('def stand: ' + str(np.std(provider)))


def count_useless_files():
    ds_path = 'ds_sequences_no_empty_properties/Benign'
    json_path_list = [join(ds_path, f) for f in listdir(ds_path) if isfile(join(ds_path, f))]
    activity_at_zero = 0
    all_at_zero = 0
    i = 0
    for json_path in json_path_list:
        with open(json_path) as f:
            json_example = json.load(f)
            f.close()
            if len(json_example['activity']) == 0:
                activity_at_zero += 1

            if len(json_example['activity']) == 0 and len(json_example['receiver']) == 0 and len(
                    json_example['service']) == 0 and len(json_example['provider']) == 0:
                all_at_zero += 1

        i += 1
        if i % 100 == 0:
            print(str(i) + ' examples computed')

    print("Activity at zero: " + str(activity_at_zero))
    print("All at zero: " + str(all_at_zero))


def count_level_two_metrics(ds_path_mal, ds_path_ben=None, spec=None):
    json_path_list = [join(ds_path_mal, f) for f in listdir(ds_path_mal) if isfile(join(ds_path_mal, f))]
    if ds_path_ben is not None:
        json_path_list = json_path_list + [join(ds_path_ben, f) for f in listdir(ds_path_ben) if
                                           isfile(join(ds_path_ben, f))]
    print('Number of examples to compute: ' + str(len(json_path_list)))

    level_two_lengths = []
    level_two_complete_string_frequency_max = []
    level_two_complete_string_frequency_mean = []
    level_two_complete_string_repetitions = []
    number_of_sequences_for_each_example = []
    total_dataset_strings = dict()
    complete_string_for_each_subgraph = []
    examples_with_no_sequences = 0
    total_sequences = 0
    sequences_with_repetition = 0
    i = 0
    for json_path in json_path_list:
        with open(json_path) as f:
            json_example = json.load(f)
            f.close()
            if (len(json_example['activity']) + len(json_example['receiver']) +
               len(json_example['service']) + len(json_example['provider'])) == 0:
                examples_with_no_sequences += 1
            else:
                number_of_sequences_for_each_example.append(
                    len(json_example['activity']) + len(json_example['receiver'])
                    + len(json_example['service']) + len(json_example['provider']))
                for macro_category in json_example.keys():
                    for level_one_action in json_example[macro_category].keys():
                        level_two_sequence = json_example[macro_category][level_one_action]
                        total_sequences += 1

                        complete_string_counter = Counter(level_two_sequence)

                        complete_string_for_each_subgraph.append(len(complete_string_counter))

                        level_two_lengths.append(len(json_example[macro_category][level_one_action]))
                        level_two_complete_string_frequency_mean.append(np.mean(list(complete_string_counter.values())))
                        level_two_complete_string_frequency_max.append(np.max(list(complete_string_counter.values())))

                        level_two_complete_string_repetitions.append(len([repetitions for repetitions
                                                                          in complete_string_counter.values()
                                                                          if repetitions > 1]))
                        if any(repetition > 1 for repetition in complete_string_counter.values()):
                            sequences_with_repetition += 1

                        total_dataset_strings = {k: total_dataset_strings.get(k, 0) + complete_string_counter.get(k, 0)
                                                 for k in set(total_dataset_strings) | set(complete_string_counter)}

        i += 1
        if i % 100 == 0:
            print(str(i) + ' examples computed')

    '''with open('mean_frequencies_complete_string.npy', 'wb') as f:
        np.save(f, np.array(level_two_complete_string_frequency_mean))
        f.close()
    with open('max_frequencies_complete_string.npy', 'wb') as f:
        np.save(f, np.array(level_two_complete_string_frequency_max))
        f.close()
    with open('level_two_sequences_lengths.npy', 'wb') as f:
        np.save(f, np.array(level_two_lengths))
        f.close()
    with open(spec + '_total_dataset_strings.pkl', 'wb') as f:
        pickle.dump(list(total_dataset_strings.keys()), f)
        f.close()'''

    newDict = {'methods': total_dataset_strings.keys(), 'frequencies': total_dataset_strings.values()}
    df = pd.DataFrame(newDict).sort_values('frequencies', ascending=False)
    df.to_excel('result.xlsx')

    print("total sequences: " + str(total_sequences))
    print("total sequences with at least one repetition: " + str(sequences_with_repetition))
    print("mean sequences length: " + str(np.mean(level_two_lengths)))
    print("mean max frequency complete string: " + str(np.mean(level_two_complete_string_frequency_max)))
    print("mean mean frequency complete string: " + str(np.mean(level_two_complete_string_frequency_mean)))
    print("max repetitions complete string: " + str(np.max(level_two_complete_string_repetitions)))
    print("mean repetitions complete string: " + str(np.mean(level_two_complete_string_repetitions)))
    print("standard deviation of repetitions complete string: " + str(np.std(level_two_complete_string_repetitions)))
    print("number of examples with no sequences: " + str(examples_with_no_sequences))

    print("mean number of sequences for each example: " + str(np.mean(number_of_sequences_for_each_example)))
    print("max number of sequences for each example: " + str(np.max(number_of_sequences_for_each_example)))

    print("max number of nodes with complete string: " + str(np.max(complete_string_for_each_subgraph)))
    print("min number of nodes with complete string: " + str(np.min(complete_string_for_each_subgraph)))
    print("mean number of nodes with complete string: " + str(np.mean(complete_string_for_each_subgraph)))
    print("standard deviation number of nodes with complete string: " + str(np.std(complete_string_for_each_subgraph)))

    sorted_ = np.sort(complete_string_for_each_subgraph)[::-1]
    generate_line_plot(sorted_, "Complete string nodes for each subgraph " + str(spec) + "top tf-idf",
                       "complete_string_class_plot_" + str(spec) + "_top_tfidf.jpg",
                       "Sequence number", "graph's nodes")
    generate_boxplot(sorted_, "Complete string nodes for each subgraph " + str(spec) + "top tf-idf",
                     "complete_string_boxplot_" + str(spec) + "_top_tfidf.jpg")
    generate_boxplot(number_of_sequences_for_each_example, "Number of sequences for each example " + str(spec)
                     + " top tf-idf", "Number_of_sequences_for_each_example_" + str(spec) + "_top_tf-idf")
    generate_boxplot(level_two_complete_string_repetitions, "Number of repetitions for each sequence " + str(spec)
                     + " top tf-idf", "Number_of_repetitions_for_each_sequence_" + str(spec) + "_top_tf-idf")


def split_train_and_test():
    ds_path = 'ds_sequences_no_empty_activities/Malicious'
    json_path_list = [f for f in listdir(ds_path) if isfile(join(ds_path, f))]
    x_train, x_test = train_test_split(json_path_list, test_size=0.2, random_state=42)
    for filename in x_train:
        os.rename("ds_sequences_no_empty_activities/Malicious/" + filename,
                  "ds_sequences_no_empty_activities/train/Malicious/" + filename)
    for filename in x_test:
        os.rename("ds_sequences_no_empty_activities/Malicious/" + filename,
                  "ds_sequences_no_empty_activities/test/Malicious/" + filename)

    ds_path = 'ds_sequences_no_empty_activities/Benign'
    json_path_list = [f for f in listdir(ds_path) if isfile(join(ds_path, f))]
    x_train, x_test = train_test_split(json_path_list, test_size=0.2, random_state=42)
    for filename in x_train:
        os.rename("ds_sequences_no_empty_activities/Benign/" + filename,
                  "ds_sequences_no_empty_activities/train/Benign/" + filename)
    for filename in x_test:
        os.rename("ds_sequences_no_empty_activities/Benign/" + filename,
                  "ds_sequences_no_empty_activities/test/Benign/" + filename)


def delete_files_with_empty_properties():
    ds_path = 'ds_sequences_no_empty_properties/Benign'
    json_path_list = [join(ds_path, f) for f in listdir(ds_path) if isfile(join(ds_path, f))]
    i = 0
    for json_path in json_path_list:
        with open(json_path) as f:
            json_example = json.load(f)
            f.close()
            if len(json_example['activity']) == 0 and len(json_example['receiver']) == 0 and len(
                    json_example['service']) == 0 and len(json_example['provider']) == 0:
                os.remove(json_path)
        i += 1
        if i % 100 == 0:
            print(str(i) + ' benign examples computed')
    ds_path = 'ds_sequences_no_empty_properties/Malicious'
    json_path_list = [join(ds_path, f) for f in listdir(ds_path) if isfile(join(ds_path, f))]
    i = 0
    for json_path in json_path_list:
        with open(json_path) as f:
            json_example = json.load(f)
            f.close()
            if len(json_example['activity']) == 0 and len(json_example['receiver']) == 0 and len(
                    json_example['service']) == 0 and len(json_example['provider']) == 0:
                os.remove(json_path)
        i += 1
        if i % 100 == 0:
            print(str(i) + ' malicious examples computed')


def count_activities_classes_methods_arches(ds_path):
    json_path_list = [join(ds_path, f) for f in listdir(ds_path) if isfile(join(ds_path, f))]
    print('Number of examples to compute: ' + str(len(json_path_list)))

    total_dataset_classes = set()
    number_of_classes_for_each_graph = []

    total_dataset_methods = dict()
    number_of_methods_for_each_graph = []
    mean_of_methods_frequencies_for_each_graph = []

    number_of_edges_for_each_graph = []
    mean_of_edges_frequencies_for_each_graph = []
    i = 0
    for json_path in json_path_list:
        with open(json_path) as f:
            last_class = 'Initial'
            graph_unique_classes = set()
            graph_unique_methods = defaultdict(int)
            graph_unique_edges = defaultdict(int)
            json_example = json.load(f)
            f.close()
            for macro_category in json_example.keys():
                for level_one_action in json_example[macro_category].keys():
                    activity_class = level_one_action.split(';->')[0] if len(
                        level_one_action.split(';->')) > 1 else 'noClass'
                    activity_method = level_one_action.split(';->')[1] if len(
                        level_one_action.split(';->')) > 1 else 'noMethod'
                    activity_edge = last_class + activity_class + activity_method
                    last_class = activity_class

                    graph_unique_classes.add(activity_class)
                    graph_unique_edges[activity_edge] += 1
                    graph_unique_methods[activity_method] += 1

            total_dataset_classes = total_dataset_classes.union(graph_unique_classes)
            number_of_classes_for_each_graph.append(len(graph_unique_classes))

            number_of_edges_for_each_graph.append(len(graph_unique_edges))
            mean_of_edges_frequencies_for_each_graph.append(np.mean(np.array([*graph_unique_edges.values()])))

            total_dataset_methods = {k: total_dataset_methods.get(k, 0) + graph_unique_methods.get(k, 0) for k in set(
                total_dataset_methods) | set(graph_unique_methods)}
            number_of_methods_for_each_graph.append(len(graph_unique_methods))
            mean_of_methods_frequencies_for_each_graph.append(np.mean(np.array([*graph_unique_methods.values()])))

        i += 1
        if i % 100 == 0:
            print(str(i) + ' examples computed')
    print('\n\n\n')

    print('Total number of classes in level_one_action: ' + str(len(total_dataset_classes)))
    print('\n')
    print('Measures for classes in each graph')
    number_of_classes_for_each_graph = np.array(number_of_classes_for_each_graph)
    print('max: ' + str(np.max(number_of_classes_for_each_graph)))
    print('min: ' + str(np.min(number_of_classes_for_each_graph)))
    print('mean: ' + str(np.mean(number_of_classes_for_each_graph)))
    print('def stand: ' + str(np.std(number_of_classes_for_each_graph)))
    print('\n\n\n')

    print('Total number of methods in level_one_action: ' + str(len(total_dataset_methods)))
    print('Measures for method frequencies in total dataset')
    frequencies = np.array([*total_dataset_methods.values()])
    print('max: ' + str(np.max(frequencies)))
    print('min: ' + str(np.min(frequencies)))
    print('mean: ' + str(np.mean(frequencies)))
    print('def stand: ' + str(np.std(frequencies)))
    print('max used method: ' + str(max(total_dataset_methods, key=total_dataset_methods.get)))
    newDict = {'methods': total_dataset_methods.keys(), 'frequencies': total_dataset_methods.values()}
    df = pd.DataFrame(newDict).sort_values('frequencies', ascending=False)
    df.to_excel('Studio dataset livello due.xlsx')
    print(df)
    print('\n')

    print('Measures for methods in each graph')
    number_of_methods_for_each_graph = np.array(number_of_methods_for_each_graph)
    print('max: ' + str(np.max(number_of_methods_for_each_graph)))
    print('min: ' + str(np.min(number_of_methods_for_each_graph)))
    print('mean: ' + str(np.mean(number_of_methods_for_each_graph)))
    print('def stand: ' + str(np.std(number_of_methods_for_each_graph)))
    print('\n\n\n')

    print('Measures for method frequencies in each graph')
    mean_of_methods_frequencies_for_each_graph = np.array(mean_of_methods_frequencies_for_each_graph)
    print('max mean: ' + str(np.max(mean_of_methods_frequencies_for_each_graph)))
    print('min mean: ' + str(np.min(mean_of_methods_frequencies_for_each_graph)))
    print('mean mean: ' + str(np.mean(mean_of_methods_frequencies_for_each_graph)))
    print('def stand mean: ' + str(np.std(mean_of_methods_frequencies_for_each_graph)))
    print('\n\n\n')

    print('Measures for edges in each graph')
    number_of_edges_for_each_graph = np.array(number_of_edges_for_each_graph)
    print('max: ' + str(np.max(number_of_edges_for_each_graph)))
    print('min: ' + str(np.min(number_of_edges_for_each_graph)))
    print('mean: ' + str(np.mean(number_of_edges_for_each_graph)))
    print('def stand: ' + str(np.std(number_of_edges_for_each_graph)))
    print('\n\n\n')

    print('Measures for edge frequencies in each graph')
    mean_of_edges_frequencies_for_each_graph = np.array(mean_of_edges_frequencies_for_each_graph)
    print('max mean: ' + str(np.max(mean_of_edges_frequencies_for_each_graph)))
    print('min mean: ' + str(np.min(mean_of_edges_frequencies_for_each_graph)))
    print('mean mean: ' + str(np.mean(mean_of_edges_frequencies_for_each_graph)))
    print('def stand mean: ' + str(np.std(mean_of_edges_frequencies_for_each_graph)))
    print('\n\n\n')


def compute_tf_idf(ds_path_mal, ds_path_ben, top_n=100):
    try:
        json_path_list = ([join(ds_path_mal, f) for f in listdir(ds_path_mal) if isfile(join(ds_path_mal, f))]
                          + [join(ds_path_ben, f) for f in listdir(ds_path_ben) if isfile(join(ds_path_ben, f))])
        print('Number of examples to compute: ' + str(len(json_path_list)))

        document_list = []
        i = 0
        for json_path in json_path_list:
            with open(json_path) as f:
                json_example = json.load(f)
                f.close()
                for macro_category in json_example.keys():
                    for level_one_action in json_example[macro_category].keys():
                        document_list.append(json_example[macro_category][level_one_action])
            i += 1
            if i % 100 == 0:
                print(str(i) + ' examples computed')

        before = np.datetime64(datetime.now())
        vectorizer = TfidfVectorizer(stop_words=['Landroid'], lowercase=False, tokenizer=custom_tokenizer,
                                     dtype=np.float32)
        X_train = vectorizer.fit_transform(document_list).toarray()
        after = np.datetime64(datetime.now())
        #global_top10_idx = X_train.max(axis=0).argsort()[-50:]
        feature_names = vectorizer.get_feature_names_out()

        print("collecting results")
        # Calcola la somma dei valori TF-IDF per ogni parola
        tfidf_sums = np.asarray(X_train.sum(axis=0)).flatten()
        tfidf_means = np.asarray(X_train.mean(axis=0)).flatten()

        print("Building results dataframe")
        # Crea un DataFrame per visualizzare i risultati
        df = pd.DataFrame({
            'Word': feature_names,
            'Importance (sum)': tfidf_sums,
            'Importance (mean)': tfidf_means
        })

        print("Creating results excell")
        # Ordina il DataFrame in base all'importanza delle parole in ordine decrescente
        df_sorted = df.sort_values(by='Importance (mean)', ascending=False)
        df_sorted.head(top_n).to_excel('tfidf.xlsx')
        #print(np.asarray(vectorizer.get_feature_names_out())[global_top10_idx])
        print(X_train)
        print(vectorizer.get_feature_names_out(), len(vectorizer.get_feature_names_out()))
        print(convertTimeDelta(before, after))
    except EOFError:
        print("Il file è vuoto o danneggiato.")
    except pickle.UnpicklingError:
        print("Errore durante il caricamento del file.")
    except FileNotFoundError:
        print("Il file non è stato trovato.")


def custom_tokenizer(text):
    # Tokenizzazione di base basata su spazi, ma includi i simboli $
    #text = text.replace(';->', ' ')
    #return re.findall(r'\b\w+\$[\w]*\b|\b\w+\b', text)
    return text


def convertTimeDelta(before, after):
    tot = after - before
    d = np.timedelta64(tot, 'D')
    tot -= d
    h = np.timedelta64(tot, 'h')
    tot -= h
    m = np.timedelta64(tot, 'm')
    tot -= m
    s = np.timedelta64(tot, 's')
    tot -= s
    ms = np.timedelta64(tot, 'ms')
    out = str(d) + " " + str(h) + " " + str(m) + " " + str(s) + " " + str(ms)
    return out
