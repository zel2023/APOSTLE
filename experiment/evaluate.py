import pickle
import torch
import matplotlib.pyplot as plt
import numpy as np
import Levenshtein
import math
import pandas as pd
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
import os
from representation.word2vector import Word2vector
import json
import sklearn.metrics as metrics
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix, average_precision_score
from experiment.ML4prediction import MlPrediction
from tqdm import tqdm
import seaborn as sns
from matplotlib.patches import PathPatch
import scipy.stats as stats
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import time

notRecognizedByBert = ['Correct-Lang-6-patch1', 'Correct-Lang-6-patch1', 'Correct-Lang-6-patch1',
                       'Correct-Lang-6-patch1', 'Correct-Lang-6-patch1', 'Correct-Lang-6-patch1',
                       'Correct-Lang-6-patch1', 'Correct-Lang-6-patch1', 'Correct-Lang-6-patch1',
                       'Correct-Lang-25-patch1', 'Correct-Lang-53-patch1', 'Incorrect-Math-6-patch2',
                       'Incorrect-Math-6-patch2', 'Incorrect-Math-6-patch1', 'Correct-Math-56-patch1',
                       'Incorrect-Math-80-patch1', 'Incorrect-Math-104-patch1']
notRecognizedByCC2Vec = ['Correct-Lang-25-patch1', 'Correct-Lang-53-patch1', 'Correct-Math-56-patch1',
                         'Incorrect-Math-80-patch1']
notRecognized = notRecognizedByBert + notRecognizedByCC2Vec

MODEL_MODEL_LOAD_PATH = '../models/java14_model/saved_model_iter8.release'
MODEL_CC2Vec = '../representation/CC2Vec/'


class evaluation:
    def __init__(self, patch_w2v, test_w2v, test_data, test_name, test_vector, patch_vector, exception_type,thre1,thre2):
        self.patch_w2v = patch_w2v
        self.test_w2v = test_w2v
        self.test_data = test_data

        self.test_name = test_name
        # self.patch_name = None
        self.test_vector = test_vector
        self.patch_vector = patch_vector
        self.exception_type = exception_type

        self.thre1=thre1
        self.thre2=thre2

        if patch_w2v == 'codebert':
            self.codebert_w2v = Word2vector(patch_w2v=self.patch_w2v, thre1=self.thre1,thre2=self.thre2)
        if patch_w2v == 'unixcoder':
            self.unixcoder_w2v = Word2vector(patch_w2v=self.patch_w2v, thre1=self.thre1,thre2=self.thre2)
        if patch_w2v == 'graphcodebert':
            self.graphcodebert_w2v = Word2vector(patch_w2v=self.patch_w2v, thre1=self.thre1,thre2=self.thre2)

    def sigmoid(self, x):  # 激活函数
        return 1 / (1 + math.exp(-x))

    def find_path_patch(self, path_patch_sliced, project_id):
        available_path_patch = []

        project = project_id.split('_')[0]
        id = project_id.split('_')[1]

        tools = os.listdir(path_patch_sliced)
        for label in ['Correct', 'Incorrect']:
            for tool in tools:
                path_bugid = os.path.join(path_patch_sliced, tool, label, project, id)
                if os.path.exists(path_bugid):
                    patches = os.listdir(path_bugid)
                    for p in patches:
                        path_patch = os.path.join(path_bugid, p)
                        if os.path.isdir(path_patch):
                            available_path_patch.append(path_patch)
        return available_path_patch

    def find_path_patch_for_naturalness(self, path_patch_sliced, project_id):
        available_path_patch = []
        target_project = project_id.split('_')[0]
        target_id = project_id.split('_')[1]

        datasets = os.listdir(path_patch_sliced)
        for dataset in datasets:
            path_dataset = os.path.join(path_patch_sliced, dataset)
            benchmarks = os.listdir(path_dataset)
            for benchmark in benchmarks:
                path_benchmark = os.path.join(path_dataset, benchmark)
                tools = os.listdir(path_benchmark)
                for tool in tools:
                    path_tool = os.path.join(path_benchmark, tool)
                    labels = os.listdir(path_tool)
                    for label in labels:
                        path_label = os.path.join(path_tool, label)
                        projects = os.listdir(path_label)
                        for project in projects:
                            if project != target_project:
                                continue
                            path_project = os.path.join(path_label, project)
                            ids = os.listdir(path_project)
                            for id in ids:
                                if id != target_id:
                                    continue
                                path_id = os.path.join(path_project, id)
                                patches = os.listdir(path_id)
                                for patch in patches:
                                    # parse patch
                                    if label == 'Correct':
                                        label_int = 1
                                    elif label == 'Incorrect':
                                        label_int = 0
                                    else:
                                        raise
                                    path_single_patch = os.path.join(path_id, patch)
                                    if os.path.isdir(path_single_patch):
                                        available_path_patch.append(path_single_patch)

                                    # for root, dirs, files in os.walk(path_single_patch):
                                    #     for file in files:
                                    #         if file.endswith('.patch'):
                                    #             try:
                                    #                 with open(os.path.join(root, file), 'r+') as f:
                                    #                     patch_diff += f.readlines()
                                    #             except Exception as e:
                                    #                 print(e)
                                    #                 continue

        return available_path_patch

    def engineered_features(self, path_json):
        other_vector = []
        P4J_vector = []
        repair_patterns = []
        repair_patterns2 = []
        try:
            with open(path_json, 'r') as f:
                feature_json = json.load(f)
                features_list = feature_json['files'][0]['features']
                P4J = features_list[-3]
                RP = features_list[-2]
                RP2 = features_list[-1]

                '''
                # other
                for k,v in other.items():
                    # if k.startswith('FEATURES_BINARYOPERATOR'):
                    #     for k2,v2 in other[k].items():
                    #         for k3,v3 in other[k][k2].items():
                    #             if v3 == 'true':
                    #                 other_vector.append('1')
                    #             elif v3 == 'false':
                    #                 other_vector.append('0')
                    #             else:
                    #                 other_vector.append('0.5')
                    if k.startswith('S'):
                        if k.startswith('S6'):
                            continue
                        other_vector.append(v)
                    else:
                        continue
                '''

                # P4J
                if not list(P4J.keys())[100].startswith('P4J'):
                    raise
                for k, v in P4J.items():
                    # dict = P4J[i]
                    # value = list(dict.values())[0]
                    P4J_vector.append(int(v))

                # repair pattern
                for k, v in RP['repairPatterns'].items():
                    repair_patterns.append(v)

                # repair pattern 2
                for k, v in RP2.items():
                    repair_patterns2.append(v)

                # for i in range(len(features_list)):
                #     dict_fea = features_list[i]
                #     if 'repairPatterns' in dict_fea.keys():
                #             # continue
                #             for k,v in dict_fea['repairPatterns'].items():
                #                 repair_patterns.append(int(v))
                #     else:
                #         value = list(dict_fea.values())[0]
                #         engineered_vector.append(value)
        except Exception as e:
            print('name: {}, exception: {}'.format(path_json, e))
            return []

        if len(P4J_vector) != 156 or len(repair_patterns) != 26 or len(repair_patterns2) != 13:
            print('name: {}, exception: {}'.format(path_json, 'null feature or shape error'))
            return []

        return P4J_vector + repair_patterns


    def vector4patch(self, available_path_patch, compare=True, ):
        vector_list = []
        vector_ML_list = []
        label_list = []
        error_list = []
        error_list2 = []
        patch_list = []
        name_list = []
        patch_id_list = []
        for p in available_path_patch:
            recogName = '-'.join([p.split('/')[-4], p.split('/')[-3], p.split('/')[-2], p.split('/')[-1]])
            if recogName in notRecognized:  # some specific patches can not be recognized
                continue

            # vector
            json_key = p + '_.json'  # pre-saved bert vector of apostle
            json_key_cross = p + '_cross.json'  # pre-saved bert feature in Haoye's ASE2020

            if self.patch_w2v == 'codebert':
                # vector, _ = self.codebert_w2v.convert_single_patch(p)
                vector, error, error2, _ = self.codebert_w2v.convert_single_patch(p)
            elif self.patch_w2v == 'unixcoder':
                vector, error, error2, _ = self.unixcoder_w2v.convert_single_patch(p)
            elif self.patch_w2v == 'graphcodebert':
                vector, error, error2, _ = self.graphcodebert_w2v.convert_single_patch(p)
            else:
                raise
            # if list(vector.astype(float)) == list(np.zeros(240).astype(float)) or list(vector.astype(float)) == list(np.zeros(1024).astype(float)):
            #     ttt = '-'.join([p.split('/')[-4], p.split('/')[-3], p.split('/')[-2], p.split('/')[-1]])
            #     notRecognized.append(ttt)
            vector_list.append(vector)
            error_list.append(error)
            error_list2.append(error2)
            # patch_list.append(patch)
            # compared with Haoye's ASE2020
            if compare:
                # Bert
                with open(json_key_cross, 'r+') as f:
                    vector_str = json.load(f)
                    vector_ML = np.array(list(map(float, vector_str)))
                vector_ML_list.append(vector_ML)

            # label
            if 'Correct' in p:
                label_list.append(1)
                label = 'Correct'
            elif 'Incorrect' in p:
                label_list.append(0)
                label = 'Incorrect'
            else:
                raise Exception('wrong label')

            # name for apostle project
            tool = p.split('/')[-5]
            patch = p.split('/')[-1]
            # name = tool + '-' + label + '-' + patchid
            # name = tool[:3] + patchid.replace('patch','')
            name = tool + patch.replace('patch', '')
            name_list.append(name)

            # patch id for Naturalness project
            patch = p.split('/')[-1]
            project_id = p.split('/')[-3] + '-' + p.split('/')[-2]
            tool = p.split('/')[-5]
            dataset = p.split('/')[-7]
            patch_id = patch + '-' + project_id + '_' + tool + '_' + dataset
            patch_id_list.append(patch_id)
        return patch_id_list, label_list, vector_list, vector_ML_list, error_list, error_list2

    # vector4patch_patchsim没用到
    def vector4patch_patchsim(self, available_path_patch, compare=True, ):
        vector_list = []
        vector_ML_list = []
        label_list = []
        name_list = []
        for p in available_path_patch:

            # vector
            json_key = p + '_.json'
            json_key_cross = p + '_cross.json'

            if self.patch_w2v == 'codebert':
                vector, _ = self.codebert_w2v.convert_single_patch(p)
            elif self.patch_w2v == 'unixcoder':
                vector, _ = self.unixcoder_w2v.convert_single_patch(p)
            elif self.patch_w2v == 'graphcodebert':
                vector, _ = self.graphcodebert_w2v.convert_single_patch(p)
            else:
                raise
            # if list(vector.astype(float)) == list(np.zeros(240).astype(float)) or list(vector.astype(float)) == list(np.zeros(1024).astype(float)):
            #     ttt = '-'.join([p.split('/')[-4], p.split('/')[-3], p.split('/')[-2], p.split('/')[-1]])
            #     notRecognized.append(ttt)
            vector_list.append(vector)
            if compare:
                with open(json_key_cross, 'r+') as f:
                    vector_str = json.load(f)
                    vector_ML = np.array(list(map(float, vector_str)))
                vector_ML_list.append(vector_ML)

            # label
            if 'Correct' in p:
                label_list.append(1)
                label = 'Correct'
            elif 'Incorrect' in p:
                label_list.append(0)
                label = 'Incorrect'
            else:
                raise Exception('wrong label')

            # name
            tool = p.split('/')[-5]
            patchid = p.split('/')[-1]
            # name = tool + '-' + label + '-' + patchid
            # name = tool[:3] + patchid.replace('patch','')
            name = '-'.join([tool[:3], p.split('/')[-4], p.split('/')[-3], p.split('/')[-2], patchid])
            name_list.append(name)

        return name_list, label_list, vector_list, vector_ML_list

    # baseline
    def get_correct_patch_list(self, failed_test_index, model=None):
        scaler = Normalizer()
        all_test_vector = scaler.fit_transform(self.test_vector)

        scaler_patch = None
        if model == 'string':
            all_patch_vector = self.patch_vector
        else:
            scaler_patch = scaler.fit(self.patch_vector)
            all_patch_vector = scaler_patch.transform(self.patch_vector)

        # construct new test and patch dataset(repository) by excluding the current failed test cases being predicted
        dataset_test = np.delete(all_test_vector, failed_test_index, axis=0)
        dataset_patch = np.delete(all_patch_vector, failed_test_index, axis=0)
        dataset_name = np.delete(self.test_name, failed_test_index, axis=0)
        dataset_func = np.delete(self.test_data[3], failed_test_index, axis=0)
        dataset_exp = np.delete(self.exception_type, failed_test_index, axis=0)

        return dataset_patch


    def get_associated_patch_list(self, failed_test_index, k=5, cut_off=0.0, model=None):
        scaler = Normalizer()
        all_test_vector = scaler.fit_transform(self.test_vector)

        scaler_patch = None
        if model == 'string':
            all_patch_vector = self.patch_vector
        else:
            scaler_patch = scaler.fit(self.patch_vector)
            all_patch_vector = scaler_patch.transform(self.patch_vector)

        # construct new test and patch dataset(repository) by excluding the current failed test cases being predicted
        dataset_test = np.delete(all_test_vector, failed_test_index, axis=0)
        dataset_patch = np.delete(all_patch_vector, failed_test_index, axis=0)
        dataset_name = np.delete(self.test_name, failed_test_index, axis=0)
        dataset_func = np.delete(self.test_data[3], failed_test_index, axis=0)
        dataset_exp = np.delete(self.exception_type, failed_test_index, axis=0)

        patch_list = []  # the associated patches of similar test cases
        cut_off_list = []
        closest_score = []
        for i in failed_test_index:
            failed_test_vector = all_test_vector[i]

            # Deprecated. exception type of current bug id.
            exp_type = self.exception_type[i]
            if ':' in exp_type:
                exp_type = exp_type.split(':')[0]

            score_test = []
            # find the k most closest test vector from other bug-id
            for j in range(len(dataset_test)):
                simi_test_vec = dataset_test[j]

                # Deprecated. exception type from other bug-id.
                simi_exp_type = dataset_exp[j]
                if ':' in simi_exp_type:
                    simi_exp_type = simi_exp_type.split(':')[0]
                flag = 1 if exp_type == simi_exp_type else 0

                dist = distance.euclidean(simi_test_vec, failed_test_vector) / (
                            1 + distance.euclidean(simi_test_vec, failed_test_vector))
                score_test.append([j, 1 - dist, flag])  # we use similarity instead of distance
            k_index_list = sorted(score_test, key=lambda x: float(x[1]), reverse=True)[:k]
            closest_score.append(k_index_list[0][1])
            # print('the closest test score is {}'.format(k_index_list[0][1]))

            # keep the test case with simi score >= 0.8 or *
            k_index = np.array([v[0] for v in k_index_list if v[1] >= cut_off])
            cut_offs = np.array([v[1] for v in k_index_list if v[1] >= cut_off])

            if k_index.size == 0:
                continue

            # exhibit the similar test case
            print('******')
            print('{}'.format(self.test_name[i]))
            print('the similar test cases:')
            k_simi_test = dataset_name[k_index]
            func = dataset_func[k_index]
            for t in range(len(k_simi_test)):
                print('{}'.format(k_simi_test[t]))
                # print('{}'.format(func[t]))

            k_patch_vector = dataset_patch[k_index]
            patch_list.append(k_patch_vector)
            cut_off_list.append(cut_offs)

            # print('exception type: {}'.format(exp_type.split('.')[-1]))
        return patch_list, scaler_patch, closest_score, cut_off_list

    def cluster_getk_test_case(self, failed_test_index, k=5, cut_off=0.0, model=None):
        scaler = Normalizer()
        all_test_vector = scaler.fit_transform(self.test_vector)
        scaler_patch = None
        if model == 'string':
            all_patch_vector = self.patch_vector
        else:
            scaler_patch = scaler.fit(self.patch_vector)
            all_patch_vector = scaler_patch.transform(self.patch_vector)

        # construct new test and patch dataset(repository) by excluding the current failed test cases being predicted
        dataset_test = np.delete(all_test_vector, failed_test_index, axis=0)
        dataset_patch = np.delete(all_patch_vector, failed_test_index, axis=0)
        dataset_name = np.delete(self.test_name, failed_test_index, axis=0)
        dataset_func = np.delete(self.test_data[3], failed_test_index, axis=0)
        dataset_exp = np.delete(self.exception_type, failed_test_index, axis=0)

        patch_list = []  # the associated patches of similar test cases
        cut_off_list = []
        closest_score = []

        # for i in range(0,len(dataset_test)):
        #     dict[dataset_test[i]]=i

        dataset_copy = dataset_test

        km = KMeans(n_clusters=10).fit(dataset_test)
        print(len(dataset_test))
        for i in failed_test_index:
            failed_test_vector = all_test_vector[i]

            # Deprecated. exception type of current bug id.
            exp_type = self.exception_type[i]
            if ':' in exp_type:
                exp_type = exp_type.split(':')[0]

            score_test = []
            for j in range(0, 10):
                dist = distance.euclidean(km.cluster_centers_[j], failed_test_vector)
                dist = 1 - dist / (1 + dist)
                score_test.append([j, dist])  # we use similarity instead of distance
            # 得到最相似的聚类中心的下标集合center_index
            maxx = 0
            center_index = -1
            # print(score_test)
            for v in score_test:
                if v[1] >= cut_off and v[1] > maxx:
                    center_index = v[0]
                    maxx = v[1]
            # 通过center—_index找到聚类中标签和它一致的所有向量

            if center_index == -1:
                continue

            # print("have come hereeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
            # k_simi_test = dataframe[(km.labels_ == center_index)]
            resSeries = pd.Series(km.labels_)
            k_index = resSeries[resSeries.values == center_index]
            k_index = k_index.index.tolist()
            # k_index=[]

            # for t in range(0,len(k_simi_test)):
            #     k_index.append(dict[k_simi_test[t]])
            # k_index=np.array(k_index)
            # print(center_index)

            # print(k_simi_test)
            # print(dataset_copy)
            # for p in k_simi_test:
            #    for q in range(0,len(dataset_copy)):
            #        if p == dataset_copy[q]:
            #            k_index.append(q)
            #            break

            # exhibit the similar test case
            print('******')
            print('{}'.format(self.test_name[i]))
            print('the similar test cases:')
            k_simi_test = dataset_name[k_index]
            for t in range(len(k_simi_test)):
                print('{}'.format(k_simi_test[t]))

            k_patch_vector = dataset_patch[k_index]
            patch_list.append(k_patch_vector)
            # cut_off_list.append(cut_offs)

            # print('exception type: {}'.format(exp_type.split('.')[-1]))
        return patch_list, scaler_patch,

    def predict_collected_projects(self, path_collected_patch=None, cut_off=0.8, distance_method=distance.cosine,
                                   ASE2020=False, patchsim=False, method=2,cof1=0.55,cof2=0.95,):
        print('Evaluate the performance of APOSTLE')
        projects = {'Chart': 26, 'Lang': 65, 'Math': 106, 'Time': 27}
        # projects = {'Chart': 26, 'Lang': 65, 'Time': 27, 'Closure': 176, 'Math': 106, 'Cli': 40, 'Codec': 18, 'Compress': 47, 'Collections': 28,  'JacksonCore': 26, 'JacksonDatabind': 112, 'JacksonXml': 6, 'Jsoup': 93, 'Csv': 16, 'Gson': 18, 'JxPath': 22, 'Mockito': 38}
        y_preds, y_trues = [], []
        y_preds_baseline, y_trues = [], []
        MAP, MRR, number_patch_MAP = [], [], 0
        MAP_baseline, MRR_baseline, number_patch_MAP_baseline = [], [], 0
        recommend_list_project = []
        x_train, y_train, x_test, y_test = [], [], [], []
        box_projecs_co, box_projecs_inco, projects_name = [], [], []
        mean_stand_dict = {0.0: [443, 816], 0.5: ['', ''], 0.6: [273, 246], 0.7: [231, 273], 0.8: [180, 235],
                           0.9: [130, 130]}
        print('test case similarity cut-off: {}'.format(cut_off))
        test_case_similarity_list, patch1278_list_short = [], []
        patch_available_distribution = {}
        patch1278_list = []
        APOSTLE_RESULT_dict = {}
        cnt1 = 0
        cnt2 = 0
        all_name = []
        yes, no = 0, 0
        item1, item2 = [], []
        sco1, sco2 = [], []
        len_dis1, len_dis2 = [], []
        p1, p2 = [], []
        for project, number in projects.items():
            print('Testing {}'.format(project))
            for id in range(1, number + 1):
                print('----------------')
                print('{}_{}'.format(project, id))
                project_id = '_'.join([project, str(id)])
                # if project_id != 'Chart_26':
                #     continue
                # extract failed test index according to bug_id
                failed_test_index = [i for i in range(len(self.test_name)) if
                                     self.test_name[i].startswith(project_id + '-')]
                if failed_test_index == []:
                    print('Couldnt find any failed test case for this bugid: {}'.format(project_id))
                    # print('{} patches skipped'.format(len(available_path_patch)))
                    continue

                # find paths of patches generated by tools
                available_path_patch = self.find_path_patch(path_collected_patch, project_id)
                # for Naturalness project
                # available_path_patch = self.find_path_patch_for_naturalness(path_collected_patch, project_id)
                if available_path_patch == []:
                    print('No generated patches of APR tools found:{}'.format(project_id))
                    continue

                # return vector according to available_path_patch
                # if patchsim:
                #     name_list, label_list, generated_patch_list, vector_ML_list = self.vector4patch_patchsim(available_path_patch, compare=ASE2020,)
                # else:
                patch_id_list, label_list, generated_patch_list, vector_ML_list, error_list, error_list2 = self.vector4patch(
                    available_path_patch, compare=ASE2020, )

                # # depulicate
                # index_to_delete = []
                # for i in range(len(vector_ML_list)):
                #     if list(vector_ML_list[i]) in unique:
                #         index_to_delete.append(i)
                #     else:
                #         unique.append(list(vector_ML_list[i]))
                # for counter, index in enumerate(index_to_delete):
                #     index = index - counter
                #     name_list.pop(index)
                #     label_list.pop(index)
                #     generated_patch_list.pop(index)
                #     vector_ML_list.pop(index)

                if patch_id_list == []:
                    print('all the patches can not be recognized')
                    continue

                # plot distribution of correct and incorrect patches
                co = label_list.count(1)
                inco = label_list.count(0)
                box_projecs_co.append(co)
                box_projecs_inco.append(inco)
                projects_name.append(project)

                # access the associated patch list(patch search space) of similar failed test cases
                associated_patch_list, scaler_patch, closest_score, cut_off_list = self.get_associated_patch_list(
                    failed_test_index, k=5, cut_off=cut_off, model=self.patch_w2v)

                # associated_patch_list, scaler_patch = self.cluster_getk_test_case(failed_test_index, k=5, cut_off=0.5, model=self.patch_w2v)
                # baseline
                correct_patches_baseline = self.get_correct_patch_list(failed_test_index, model=self.patch_w2v)
                if associated_patch_list == []:
                    print('No closest test case that satisfied with the condition of cut-off similarity')
                    print('save train data for ML model of ASE2020')
                    # comparison with ML prediction in ASE2020
                    if ASE2020 and vector_ML_list != []:
                        for i in range(len(label_list)):
                            # if list(vector_list[i].astype(float)) != list(np.zeros(240).astype(float)):
                            x_train.append(vector_ML_list[i])
                            y_train.append(label_list[i])
                    continue
                recommend_list, recommend_list_baseline = [], []
                # calculate the center of associated patches(repository)
                centers = self.dynamic_threshold2(associated_patch_list, cut_off_list, distance_method=distance_method,
                                                  sumup='mean')
                # print("centers")
                # print(centers)

                centers_baseline = [correct_patches_baseline.mean(axis=0)]
                for i in range(len(patch_id_list)):
                    # name = name_list[i]
                    name = patch_id_list[i]
                    tested_patch = generated_patch_list[i]
                    y_true = label_list[i]  # p1'
                    y_pred_prob = 0
                    y_pred = 0
                    all_name.append(name)
                    # y_pred = self.predict_label(centers, threshold_list, vector_new_patch, scaler_patch)
                    # y_pred_prob = self.predict_prob(centers, threshold_list, vector_new_patch, scaler_patch)

                    # METHOD3
                    if method == 3:
                         for j in range(1, len(associated_patch_list)):
                                     y_pred_prob1, y_pred1 = self.predict_recom(associated_patch_list[j], tested_patch, scaler_patch, mean_stand_dict[cut_off], distance_method=distance_method,)
                                     if y_pred_prob1>y_pred_prob:
                                          y_pred_prob=y_pred_prob1
                                     if y_pred1>y_pred:
                                          y_pred=y_pred1
                    # METHOD1
                    elif method == 1:
                        dis_min = distance.euclidean(associated_patch_list[0][0], tested_patch)
                        vec = associated_patch_list[0]

                        for j in range(1, len(associated_patch_list)):
                           dis = distance.euclidean(associated_patch_list[j][0], tested_patch)
                           if dis < dis_min:
                               dis_max = dis
                               vec = associated_patch_list[j]
                        y_pred_prob, y_pred = self.predict_recom(vec, tested_patch, scaler_patch,mean_stand_dict[cut_off],distance_method=distance_method, )
                    #METHOD2
                    elif method == 2:
                        y_pred_prob, y_pred = self.predict_recom(centers, tested_patch, scaler_patch,
                                                                 mean_stand_dict[cut_off],
                                                                 distance_method=distance_method, )
                        if error_list2[i] == 1:
                            y_pred_prob *= cof1

                        if error_list[i] == 1:
                            y_pred_prob *= cof2
                        if (y_pred_prob > 0.5):
                            y_pred = 1
                        else:
                            y_pred = 0
                    y_pred_prob_baseline, y_pred_baseline = self.predict_recom(centers_baseline, tested_patch,
                                                                               scaler_patch, mean_stand_dict[cut_off],
                                                                               distance_method=distance_method, )

                    # for Naturalness project
                    APOSTLE_RESULT_dict[name.lower()] = y_pred

                    if not math.isnan(y_pred_prob):
                        recommend_list.append([name, y_pred, y_true, y_pred_prob])
                        recommend_list_baseline.append([name, y_pred_baseline, y_true, y_pred_prob_baseline])

                        y_preds.append(y_pred_prob)
                        y_preds_baseline.append(y_pred_prob_baseline)
                        y_trues.append(y_true)

                        # the current patches addressing this bug. The highest test case similarity we can find for test case of this bug is in 'test_case_similarity_list'
                        # test_case_similarity_list.append(max(closest_score))

                        # ML prediction for comparison
                        if ASE2020:
                            x_test.append(vector_ML_list[i])
                            y_test.append(y_true)

                        # ML prediction for comparison

                        # # record distribution of available patches
                        # key = name[:3]+str(y_true)
                        # if key not in patch_available_distribution:
                        #     patch_available_distribution[key] = 1
                        # else:
                        #     patch_available_distribution[key] += 1

                        # save the name of 1278 patches for evaluating
                        path = available_path_patch[i]
                        patchname = path.split('/')[-1]
                        tool = path.split('/')[-5]
                        patchname_complete = '-'.join([patchname, project, str(id), tool, str(y_true)])
                        patch1278_list.append(patchname_complete)

                if not (
                        not 1 in label_list or not 0 in label_list) and recommend_list != []:  # ensure there are correct and incorrect patches in recommended list
                    AP, RR = self.evaluate_recommend_list(recommend_list)
                    if AP != None and RR != None:
                        MAP.append(AP)
                        MRR.append(RR)
                        number_patch_MAP += len(recommend_list)
                if not (
                        not 1 in label_list or not 0 in label_list) and recommend_list_baseline != []:  # ensure there are correct and incorrect patches in recommended list
                    AP, RR = self.evaluate_recommend_list(recommend_list_baseline)
                    if AP != None and RR != None:
                        MAP_baseline.append(AP)
                        MRR_baseline.append(RR)
                        number_patch_MAP_baseline += len(recommend_list_baseline)

                recommend_list_project += recommend_list

        # patch distribution
        # print(patch_available_distribution)

        # for Naturalness project
        with open('..\\data\\APOSTLE_RESULT_{}.json'.format(cut_off), 'w+') as f:
            json.dump(APOSTLE_RESULT_dict, f)

        if y_trues == [] or not 1 in y_trues or not 0 in y_trues:
            return




        # 1. independently
        if not ASE2020 and not patchsim:
            ## baseline
            if cut_off == 0.0:
                print('\nBaseline: ')
                self.evaluation_metrics(y_trues, y_preds_baseline)
                self.MAP_MRR_Mean(MAP_baseline, MRR_baseline, number_patch_MAP_baseline)
            ## APOSTLE
            print('\nAPOSTLE: ')
            recall_p, recall_n, acc, prc, rc, f1, auc_, result_APR = self.evaluation_metrics(y_trues, y_preds)
            self.MAP_MRR_Mean(MAP, MRR, number_patch_MAP)

        # 2. Compare and Combine
        if ASE2020 and cut_off > 0.0:
            print('------')
            print('Evaluating ASE2020 Performance')
            MlPrediction(x_train, y_train, x_test, y_test, y_pred_apostle=y_preds,
                         test_case_similarity_list=test_case_similarity_list, algorithm='lr', comparison=ASE2020,
                         cutoff=cut_off).predict()
            MlPrediction(x_train, y_train, x_test, y_test, y_pred_apostle=y_preds,
                         test_case_similarity_list=test_case_similarity_list, algorithm='rf', comparison=ASE2020,
                         cutoff=cut_off).predict()

        with open('./patch' + str(len(patch1278_list)) + '.txt', 'w+') as f:
            for p in patch1278_list:
                f.write(p + '\n')

        if patchsim:
            print('------')
            print('Evaluating PatchSim improvement')
            y_combine, y_combine_trues = [], []
            y_patchsim = []
            apostle_cnt = 0
            with open('patch325_result.txt', 'r+') as f_patchsim:
                for line in f_patchsim:
                    line = line.strip()
                    name_ps, prediction_ps = line.split(',')[0], line.split(',')[1]
                    i = patch1278_list.index(name_ps)
                    y_combine_trues.append(y_trues[i])
                    y_patchsim.append(float(prediction_ps))
                    if test_case_similarity_list[i] >= 0.8:
                        y_combine.append(y_preds[i])
                        apostle_cnt += 1
                    else:
                        y_combine.append(float(prediction_ps))
            print('apostle_cnt: {}, PatchSim_cnt: {}'.format(apostle_cnt, len(y_combine) - apostle_cnt))
            self.evaluation_metrics(y_combine_trues, y_patchsim)
            print('----------')
            self.evaluation_metrics(y_combine_trues, y_combine)

        '''
        if patchsim:
            print('------')
            print('Evaluating Incorrect Excluded on PatchSim')
            # [name, y_pred, y_true, y_pred_prob]
            recommend_list_project = pd.DataFrame(sorted(recommend_list_project, key=lambda x: x[3], reverse=True))
            Correct = recommend_list_project[recommend_list_project[2]==1]
            filter_out_incorrect = recommend_list_project.shape[0] - Correct[:].index.tolist()[-1] - 1

            print('Test data size: {}, Incorrect: {}, Correct: {}'.format(recommend_list_project.shape[0], recommend_list_project.shape[0]-Correct.shape[0],
                                                                          Correct.shape[0]))
            # print('Exclude incorrect: {}'.format(filter_out_incorrect))
            # print('Exclude rate: {}'.format(filter_out_incorrect/(recommend_list_project.shape[0]-Correct.shape[0])))
            # print('Excluded name: {}'.format(recommend_list_project.iloc[Correct[:].index.tolist()[-1]+1:][0].values))

            # topHalf = recommend_list_project.iloc[:Correct[:].index.tolist()[-1] + 1]
            # topHalfIncorrect = topHalf[topHalf[2] == 0][0].values
            # print('Noe excluded name: {}'.format(topHalfIncorrect))
        '''
        self.statistics_box(box_projecs_co, box_projecs_inco, projects_name)

    def predict_new_patch(self, project_id, cut_off=0.8, patch_vector=None, distance_method=distance.cosine,
                          threshold=0.5):
        mean_stand_dict = {0.0: [443, 816], 0.5: ['', ''], 0.6: [273, 246], 0.7: [231, 273], 0.8: [180, 235],
                           0.9: [130, 130]}

        print('----------------')
        # extract failed test index according to bug_id
        failed_test_index = [i for i in range(len(self.test_name)) if
                             self.test_name[i].startswith(project_id + '-')]
        if failed_test_index == []:
            print('############################################')
            print('Sorry, we cannot find any failed test case for this bug id: {}.'.format(project_id))
            print('Please try another bug id.')
            # print('{} patches skipped'.format(len(available_path_patch)))
            return

        # access the associated patch list(patch search space) of similar failed test cases
        associated_patch_list, scaler_patch, closest_score = self.get_associated_patch_list(failed_test_index,
                                                                                            k=5,
                                                                                            cut_off=cut_off,
                                                                                            model=self.patch_w2v)
        if associated_patch_list == []:
            print('############################################')
            print('Sorry, there is no any test case satisfying with the requirement of test case similarity cut-off.')
            print('Please try to decrease the cut-off.')
            return

        # calculate the center of associated patches(repository)
        centers = self.dynamic_threshold2(associated_patch_list, distance_method=distance_method, sumup='mean')

        tested_patch = patch_vector
        y_pred_prob, _ = self.predict_recom(centers, tested_patch, scaler_patch, mean_stand_dict[cut_off],
                                            distance_method=distance_method, )
        print('############################################')
        y_pred = 1 if y_pred_prob >= threshold else 0
        if y_pred == 1:
            print('Congrats! Your patch is CORRECT.')
        elif y_pred == 0:
            print('Sorry, your patch is INCORRECT.')
        print(y_pred_prob)

    # def improve_ML(self, path_collected_patch=None, cut_off=0.8, distance_method = distance.cosine, kfold=10, algorithm='lr', method='combine'):
    #     print('Research Question 3: Improvement')
    #     projects = {'Chart': 26, 'Lang': 65, 'Math': 106, 'Time': 27}
    #     y_preds_apostle, y_preds_prob_apostle, y_trues = [], [], []
    #     x_all, y_all, x_test, y_test = [], [], [], []
    #     # comparison = 'ASE2020' # will make comparison if the value equals to 'ASE2020'
    #     mean_stand_dict = {0.0: [443, 816], 0.6: [273, 246], 0.7: [231, 273], 0.8: [180, 235], 0.9: [130, 130]}
    #     print('test case similarity cut-off: {}'.format(cut_off))
    #     unique_dict = []
    #     for project, number in projects.items():
    #         print('Testing {}'.format(project))
    #         for id in range(1, number + 1):
    #             print('----------------')
    #             print('{}_{}'.format(project, id))
    #             project_id = '_'.join([project, str(id)])
    #
    #             # extract failed test index according to bug_id
    #             failed_test_index = [i for i in range(len(self.test_name)) if self.test_name[i].startswith(project_id+'-')]
    #             if failed_test_index == []:
    #                 print('Couldnt find any failed test case for this bugid: {}'.format(project_id))
    #                 # print('{} patches skipped'.format(len(available_path_patch)))
    #                 continue
    #
    #             # find paths of patches generated by tools
    #             available_path_patch = self.find_path_patch(path_collected_patch, project_id)
    #             if available_path_patch == []:
    #                 print('No generated patches of APR tools found:{}'.format(project_id))
    #                 continue
    #
    #             # return vector according to available_path_patch
    #             name_list, label_list, generated_patch_list, vector_ML_list, vector_ODS_list = self.vector4patch(available_path_patch, compare=ASE2020,)
    #             if name_list == []:
    #                 print('all the patches can not be recognized')
    #                 continue
    #
    #             # access the associated patch list(patch repository) of similar failed test cases
    #             associated_patch_list, scaler_patch, closest_score = self.get_associated_patch_list(failed_test_index, k=5, cut_off=cut_off, model=self.patch_w2v)
    #
    #             # print('save train data for ML model of ASE2020')
    #             if ASE2020 and vector_ML_list != []:
    #                 for i in range(len(vector_ML_list)):
    #                     # if list(vector_list[i].astype(float)) != list(np.zeros(240).astype(float)):
    #                     if vector_ML_list[i] in unique_dict:
    #                         continue
    #                     else:
    #                         x_all.append(vector_ML_list[i])
    #                         y_all.append(label_list[i])
    #
    #             # calculate the center of associated patches(repository)
    #             if associated_patch_list == []:
    #                 # fill value for the prediction of APOSTLE to keep it the same length as ML prediction
    #                 y_preds_apostle += [-999 for i in range(len(vector_ML_list))]
    #                 y_preds_prob_apostle += [-999 for i in range(len(vector_ML_list))]
    #                 y_trues += [i for i in label_list]
    #             else:
    #                 centers = self.dynamic_threshold2(associated_patch_list, distance_method=distance_method, sumup='mean')
    #                 for i in range(len(vector_ML_list)):
    #                     name = name_list[i]
    #                     tested_patch = generated_patch_list[i]
    #                     y_true = label_list[i]
    #                     # y_pred = self.predict_label(centers, threshold_list, vector_new_patch, scaler_patch)
    #                     # y_pred_prob = self.predict_prob(centers, threshold_list, vector_new_patch, scaler_patch)
    #                     y_pred_prob, y_pred = self.predict_recom(centers, tested_patch, scaler_patch, mean_stand_dict[cut_off], distance_method=distance_method,)
    #
    #                     if math.isnan(y_pred_prob):
    #                         y_preds_apostle.append(-999)
    #                         y_preds_prob_apostle.append(-999)
    #                         y_trues.append(y_true)
    #                     else:
    #                         y_preds_apostle.append(y_pred)
    #                         y_preds_prob_apostle.append(y_pred_prob)
    #                         y_trues.append(y_true)
    #
    #     # run cross validation for ML-based approach in ASE2020
    #     x_all_unique, y_all_unique, y_preds_prob_apostle_unique = [], [], []
    #     for i in range(len(x_all)):
    #         if list(x_all[i]) in unique_dict:
    #             continue
    #         else:
    #             unique_dict.append(list(x_all[i]))
    #             x_all_unique.append(x_all[i])
    #             y_all_unique.append(y_all[i])
    #             y_preds_prob_apostle_unique.append(y_preds_prob_apostle[i])
    #     x_all_unique = np.array(x_all_unique)
    #     y_all_unique = np.array(y_all_unique)
    #     y_preds_prob_apostle_unique = np.array(y_preds_prob_apostle_unique)
    #     skf = StratifiedKFold(n_splits=kfold, shuffle=True)
    #     accs, prcs, rcs, f1s, aucs = list(), list(), list(), list(), list()
    #     rcs_p, rcs_n = list(), list()
    #     for train_index, test_index in skf.split(x_all_unique, y_all_unique):
    #         x_train, y_train = x_all_unique[train_index], y_all_unique[train_index]
    #         x_test, y_test = x_all_unique[test_index], y_all_unique[test_index]
    #
    #         # prediction by apostle
    #         # y_pred_apostle = y_preds_apostle[test_index]
    #         y_test_pred_prob_apostle = y_preds_prob_apostle_unique[test_index]
    #
    #         # standard data
    #         scaler = StandardScaler().fit(x_train)
    #         # scaler = MinMaxScaler().fit(x_train)
    #         x_train = scaler.transform(x_train)
    #         x_test = scaler.transform(x_test)
    #
    #         print('\ntrain data: {}, test data: {}'.format(len(x_train), len(x_test)), end='')
    #
    #         clf = None
    #         if algorithm == 'lr':
    #             clf = LogisticRegression(solver='lbfgs', class_weight={1: 1},).fit(X=x_train, y=y_train)
    #         elif algorithm == 'dt':
    #             clf = DecisionTreeClassifier().fit(X=x_train, y=y_train, sample_weight=None)
    #         elif algorithm == 'rf':
    #             clf = RandomForestClassifier(class_weight={1: 1}, ).fit(X=x_train, y=y_train)
    #
    #         if method == 'combine':
    #             # combine both
    #             number_apostle = 0
    #             number_ML = 0
    #             y_pred_final = []
    #             for i in range(len(y_test)):
    #
    #                 # apply apostle first
    #                 if y_test_pred_prob_apostle[i] != -999:
    #                     number_apostle += 1
    #                     y_pred_final.append(y_test_pred_prob_apostle[i])
    #                 else:
    #                     number_ML += 1
    #                     y_test_pred_prob_ML = clf.predict_proba(x_test[i].reshape(1,-1))[:, 1]
    #                     y_pred_final.append(y_test_pred_prob_ML)
    #
    #                 # y_pred_final.append((y_test_pred_prob_apostle[i] + clf.predict_proba(x_test[i].reshape(1,-1))[:, 1])\\2.0)
    #             print('\nNumber of apostle and ML: {} {}'.format(number_apostle, number_ML))
    #         else:
    #             y_pred_final = clf.predict_proba(x_test)[:, 1]
    #
    #         # print('{}: '.format(algorithm))
    #         recall_p, recall_n, acc, prc, rc, f1, auc_, _ = self.evaluation_metrics(list(y_test), y_pred_final)
    #
    #         accs.append(acc)
    #         prcs.append(prc)
    #         rcs.append(rc)
    #         f1s.append(f1)
    #
    #         aucs.append(auc_)
    #         rcs_p.append(recall_p)
    #         rcs_n.append(recall_n)
    #
    #     print('\n{}-fold cross validation mean: '.format(kfold))
    #     print('Accuracy: {:.1f} -- Precision: {:.1f} -- +Recall: {:.1f} -- F1: {:.1f} -- AUC: {:.3f}'.format(np.array(accs).mean() * 100, np.array(prcs).mean() * 100, np.array(rcs).mean() * 100, np.array(f1s).mean() * 100, np.array(aucs).mean()))
    #     print('AUC: {:.3f}, +Recall: {:.3f}, -Recall: {:.3f}'.format(np.array(aucs).mean(), np.array(rcs_p).mean(), np.array(rcs_n).mean()))
    #

    def predict(self, patch_list, new_patch, scaler_patch):
        if self.patch_w2v != 'string':  ###############codebert
            new_patch = scaler_patch.transform(new_patch.reshape((1, -1)))
        dist_final = []
        # patch list includes multiple patches for multi failed test cases
        for y in range(len(patch_list)):
            patches = patch_list[y]
            dist_k = []
            for z in range(len(patches)):
                vec = patches[z]
                # dist = np.linalg.norm(vec - new_patch)
                if self.patch_w2v == 'string':
                    dist = Levenshtein.distance(vec[0], new_patch[0])
                    dist_k.append(dist)
                else:
                    # choose method to calculate distance
                    dist = distance.cosine(vec, new_patch)
                    # dist = distance.euclidean(vec, new_patch)/(1 + distance.euclidean(vec, new_patch))
                    dist_k.append(dist)

            dist_mean = np.array(dist_k).mean()
            dist_min = np.array(dist_k).min()

            # print('mean:{}  min:{}'.format(dist_mean, dist_min))
            dist_final.append(dist_min)

        dist_final = np.array(dist_final).mean()
        return dist_final

    def dynamic_threshold(self, patch_list):
        centers = []
        threshold_list = []
        # patch list includes multiple patches for multi failed test cases
        for y in range(len(patch_list)):
            patches = patch_list[y]

            # threshold 1: center of patch list
            center = np.array(patches).mean(axis=0)
            dist_mean = np.array([distance.cosine(p, center) for p in patches]).mean()
            # dist_mean = np.array([distance.cosine(p, center) for p in patches]).max()
            score_mean = 1 - dist_mean

            centers.append(center)
            threshold_list.append(score_mean)
        return centers, threshold_list

    def dynamic_threshold2(self, patch_list, cut_off_list, distance_method=distance.euclidean, sumup='mean'):
        # patch_list: [[top-5 patches for failed test case 1], [top-5 patches failed test case 2], [top-5 patches failed test case 3]]
        if self.patch_w2v != 'string':  #################codevert
            if len(patch_list) == 1:
                center = patch_list[0].mean(axis=0)
                # if sumup == 'mean':
                #     dist_mean = np.array([distance_method(p, center) for p in patch_list[0]]).mean()
                # elif sumup == 'max':
                #     dist_mean = np.array([distance_method(p, center) for p in patch_list[0]]).max()
            else:
                sum = 0
                # print(len(cut_off_list))
                for i in range(len(cut_off_list)):
                    # aa = cut_off_list[i]
                    # bb = [(aa[i - 1], i) for i in range(1, len(aa) + 1)]
                    # cc = sorted(bb)
                    # dd = [(cc[i - 1][0], i, cc[i - 1][1]) for i in range(1, len(aa) + 1)]
                    # ee = sorted(dd, key=lambda x: x[2])
                    # cut_off_list[i] = [(1/x[1]) for x in ee]
                    # print(cut_off_list[i])

                    for j in range(len(cut_off_list[i])):
                        cut_off_list[i][j] -= 0.8
                        sum += cut_off_list[i][j]
                    for j in range(len(cut_off_list[i])):
                        cut_off_list[i][j] /= sum
                        # print(cut_off_list[i][j])
                    sum = 0
                patches = np.zeros((1, 768))
                for i in range(0, len(patch_list)):
                    # print(len(cut_off_list[i]))
                    for j in range(len(cut_off_list[i])):
                        # print(cut_off_list[i][j])
                        # print(patch_list[i][j])
                        patches += cut_off_list[i][j] * patch_list[i][j]
                return patches
                # patches = np.concatenate((patches, patch_list[i]), axis=0)
                # center = patches.mean(axis=0)

        else:
            return patch_list

        return [center]

    def predict_label(self, centers, threshold_list, new_patch, scaler_patch, ):
        if self.patch_w2v != 'string':  ###########codebert
            new_patch = scaler_patch.transform(new_patch.reshape((1, -1)))

        vote_list = []
        # patch list includes multiple patches for multi failed test cases
        for y in range(len(centers)):
            center = centers[y]
            score_mean = threshold_list[y]

            # choose method to calculate distance
            dist_new = distance.cosine(new_patch, center)
            # dist_new = distance.euclidean(vec, new_patch)/(1 + distance.euclidean(vec, new_patch))

            score_new = 1 - dist_new

            vote_list.append(1 if score_new >= score_mean else 0)
        if vote_list.count(1) >= len(centers) / 2.0:
            return 1
        else:
            return 0

    def predict_prob(self, centers, threshold_list, new_patch, scaler_patch, distance_method=distance.euclidean):
        if self.patch_w2v != 'string':  ###########codebert
            new_patch = scaler_patch.transform(new_patch.reshape((1, -1)))

        center = centers[0]

        dist_new = distance_method(new_patch, center)

        # normalize range
        if distance_method == distance.euclidean:
            dist_new = dist_new / (1 + dist_new)
            score_prob_new = 1 - dist_new

        elif distance_method == distance.cosine:
            dist_new = dist_new / (1 + dist_new)
            score_prob_new = 1 - dist_new

        return score_prob_new

    def predict_recom(self, centers, new_patch, scaler_patch, mean_stand=None, distance_method=distance.euclidean):
        if self.patch_w2v != 'string':  ####codebert
            new_patch = scaler_patch.transform(new_patch.reshape((1, -1)))
            center = centers[0]

            dist_new = distance_method(new_patch, center)
            # dist_new = 1-distance_method(new_patch, center)[0][1] # pearson

            # normalize range
            # score_prob_new = self.sigmoid(1 - dist_new)
            dist_new = dist_new / (1 + dist_new)
            score_prob_new = 1 - dist_new

            # if score_prob_new >= score_mean:
            if score_prob_new >= 0.5:
                y_pred = 1
            else:
                y_pred = 0
            return score_prob_new, y_pred

        else:
            new_patch = new_patch[0]
            dist_new = []
            # mean distance to every patch
            for i in range(len(centers)):
                patches_top5 = centers[i]
                for p in patches_top5:
                    dist_new.append(Levenshtein.distance(new_patch, str(p)))
            dist_new = np.array(dist_new).mean()

            # (dist_new-mean)/stand
            dist_new = (dist_new - mean_stand[0]) / mean_stand[1]
            try:
                score_prob_new = self.sigmoid(-dist_new)
            except:
                print(dist_new)

            if score_prob_new >= 0.5:
                y_pred = 1
            else:
                y_pred = 0

            return score_prob_new, y_pred

    def evaluation_metrics(self, y_trues, y_pred_probs):
        fpr, tpr, thresholds = roc_curve(y_true=y_trues, y_score=y_pred_probs, pos_label=1)
        auc_ = auc(fpr, tpr)

        y_preds = [1 if p >= 0.5 else 0 for p in y_pred_probs]

        acc = accuracy_score(y_true=y_trues, y_pred=y_preds)
        prc = precision_score(y_true=y_trues, y_pred=y_preds)
        rc = recall_score(y_true=y_trues, y_pred=y_preds)
        f1 = 2 * prc * rc / (prc + rc)

        tn, fp, fn, tp = confusion_matrix(y_trues, y_preds).ravel()
        recall_p = tp / (tp + fn)
        recall_n = tn / (tn + fp)

        result = '***------------***\n'
        result += 'Evaluating AUC, F1, +Recall, -Recall\n'
        result += 'Test data size: {}, Correct: {}, Incorrect: {}\n'.format(len(y_trues), y_trues.count(1),
                                                                            y_trues.count(0))
        result += 'Accuracy: %f -- Precision: %f -- +Recall: %f -- F1: %f \n' % (acc, prc, rc, f1)
        result += 'AUC: {:.3f}, +Recall: {:.3f}, -Recall: {:.3f}'.format(auc_, recall_p, recall_n)
        # return , auc_

        print(result)
        # print('AP: {}'.format(average_precision_score(y_trues, y_pred_probs)))
        return recall_p, recall_n, acc, prc, rc, f1, auc_, result

    def evaluate_defects4j_projects(self, option1=True, option2=0.6):
        print('Research Question 1.2')
        scaler = Normalizer()
        all_test_vector = scaler.fit_transform(self.test_vector)
        scaler_patch = scaler.fit(self.patch_vector)
        all_patch_vector = scaler_patch.transform(self.patch_vector)

        projects = {'Chart': 26, 'Lang': 65, 'Time': 27, 'Closure': 176, 'Math': 106, 'Cli': 40, 'Codec': 18,
                    'Compress': 47, 'Collections': 28, 'JacksonCore': 26, 'JacksonDatabind': 112, 'JacksonXml': 6,
                    'Jsoup': 93, 'Csv': 16, 'Gson': 18, 'JxPath': 22, 'Mockito': 38}
        # projects = {'Chart': 26, 'Lang': 65, 'Time': 27, 'Math': 106, }
        all_closest_score = []
        box_plot = []
        for project, number in projects.items():
            print('Testing {}'.format(project))

            # go through all test cases
            cnt = 0
            for i in range(len(self.test_name)):
                # skip other projects while testing one project
                if not self.test_name[i].startswith(project):
                    continue
                # project = self.test_name[i].split('-')[0].split('_')[0]
                id = self.test_name[i].split('-')[0].split('_')[1]
                print('{}'.format(self.test_name[i]))
                this_test = all_test_vector[i]
                this_patch = all_patch_vector[i]

                # find the closest test case
                dist_min_index = None
                dist_min = np.inf
                for j in range(len(all_test_vector)):
                    # skip itself
                    if j == i:
                        continue
                    # option 1: whether skip current project-id
                    if option1 and self.test_name[j].startswith(project + '_' + id + '-'):
                        continue
                    dist = distance.euclidean(this_test, all_test_vector[j]) / (
                                1 + distance.euclidean(this_test, all_test_vector[j]))
                    if dist < dist_min:
                        dist_min = dist
                        dist_min_index = j
                sim_test = 1 - dist_min
                all_closest_score.append(sim_test)
                # option 2: threshold for test cases similarity
                if sim_test >= option2:
                    # find associated patches similarity
                    print('the closest test: {}'.format(self.test_name[dist_min_index]))
                    closest_patch = all_patch_vector[dist_min_index]
                    # distance_patch = distance.euclidean(closest_patch, this_patch)/(1 + distance.euclidean(closest_patch, this_patch))
                    distance_patch = distance.cosine(closest_patch, this_patch) / (
                                1 + distance.cosine(closest_patch, this_patch))
                    score_patch = 1 - distance_patch
                    if math.isnan(score_patch):
                        continue
                    box_plot.append([project, 'H', score_patch])

                # find average patch similarity
                simi_patch_average = []
                for p in range(len(all_patch_vector)):
                    if p == i:
                        continue
                    # dist = distance.euclidean(this_patch, all_patch_vector[p]) / (1 + distance.euclidean(this_patch, all_patch_vector[p]))
                    dist = distance.cosine(this_patch, all_patch_vector[p]) / (
                                1 + distance.cosine(this_patch, all_patch_vector[p]))
                    simi_patch = 1 - dist
                    if math.isnan(simi_patch):
                        continue
                    simi_patch_average.append(simi_patch)
                box_plot.append([project, 'N', np.array(simi_patch_average).mean()])

    def evaluate_recommend_list(self, recommend_list):
        # recommend_list: [name, y_pred, y_true, y_pred_prob]
        recommend_list = pd.DataFrame(sorted(recommend_list, key=lambda x: x[3], reverse=True),
                                      columns=['name', 'y_pred', 'y_true',
                                               'y_pred_prob'])  # rank by prediction probability

        number_correct = 0.0
        precision_all = 0.0
        for i in range(recommend_list.shape[0]):
            if recommend_list.loc[i]['y_true'] == 1:
                number_correct += 1.0
                precision_all += (number_correct / (i + 1))

        if number_correct == 0.0:
            print('No correct patch found on the recommended list')
            return None, None
        else:
            AP = precision_all / number_correct
            RR = 1.0 / (list(recommend_list[:]['y_true']).index(1) + 1)

        print('AP: {}'.format(AP))
        print('RR: {}'.format(RR))

        return AP, RR

    def MAP_MRR_Mean(self, MAP, MRR, number_patch_MAP):
        print('------')
        print('Evaluating MAP, MRR on Recommended List')
        print('Patch size: {}'.format(number_patch_MAP))
        print('Bug project size: {}'.format(len(MAP)))
        print('MAP: {}, MRR: {}'.format(np.array(MAP).mean(), np.array(MRR).mean()))

    def statistics_box(self, box_projecs_co, box_projecs_inco, projects_name):
        data = {
            'Correct': box_projecs_co,
            'Incorrect': box_projecs_inco,
            'Project': projects_name
        }

    def adjust_box_widths(self, g, fac):
        """
        Adjust the widths of a seaborn-generated boxplot.
        """

        # iterating through Axes instances
        for ax in g.axes:
            # iterating through axes artists:
            for c in ax.get_children():

                # searching for PathPatches
                if isinstance(c, PathPatch):
                    # getting current width of box:
                    p = c.get_path()
                    verts = p.vertices
                    verts_sub = verts[:-1]
                    xmin = np.min(verts_sub[:, 0])
                    xmax = np.max(verts_sub[:, 0])
                    xmid = 0.5 * (xmin + xmax)
                    xhalf = 0.5 * (xmax - xmin)

                    # setting new width of box
                    xmin_new = xmid - fac * xhalf
                    xmax_new = xmid + fac * xhalf
                    verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
                    verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new

                    # setting new width of median line
                    for l in ax.lines:
                        if np.all(l.get_xdata() == [xmin, xmax]):
                            l.set_xdata([xmin_new, xmax_new])

    def obtain_vector_for_new_patch(self, w2v, path_patch_snippet):
        w2v = Word2vector(test_w2v=None, patch_w2v=w2v, path_patch_root=path_patch_snippet)
        patch_vector, _ = w2v.convert_single_patch(path_patch_snippet)
        return patch_vector












