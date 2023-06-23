import sys, os

sys.path.append(os.path.abspath(os.path.join('..', '.')))
# import seaborn as sns
import pickle
from experiment.config import Config
from representation.word2vector import Word2vector
import numpy as np
from scipy.spatial import distance
from experiment import patch_bert_vector
from experiment.evaluate import evaluation
from experiment.cluster import cluster


class Experiment:
    def __init__(self, path_test, path_patch_root, path_generated_patch, organized_dataset, patch_w2v, test_w2v,cof1,cof2,thre1,thre2):
        self.path_test = path_test
        # self.path_test_vector = path_test_vector
        # self.path_patch_vector = path_patch_vector
        self.path_patch_root = path_patch_root
        self.path_generated_patch = path_generated_patch

        self.organized_dataset = organized_dataset
        self.patch_w2v = patch_w2v
        self.test_w2v = test_w2v

        self.original_dataset = None
        # self.patch_data = None

        self.test_name = None
        self.patch_name = None
        self.test_vector = None
        self.patch_vector = None
        self.exception_type = None
        
        self.cof1 = cof1
        self.cof2 = cof2
        self.thre1 = thre1
        self.thre2 = thre2

    # 初始化，加载test case，patch
    def load_test(self, ):
        # load original data
        # 4 times in sum
        with open(self.path_test, 'rb') as f:
            self.original_dataset = pickle.load(f)

        # organize data and vector
        if os.path.exists(organized_dataset):
            datasets = pickle.load(open(self.organized_dataset, 'rb'))
            with open(self.path_test, 'rb') as f:
                self.original_dataset = pickle.load(f)
            self.test_name = datasets[0]
            self.patch_name = datasets[1]
            self.test_vector = datasets[2]
            self.patch_vector = datasets[3]
            self.exception_type = datasets[4]  # 报错信息（异常类型）
        else:
            with open(self.path_test, 'rb') as f:
                self.original_dataset = pickle.load(f)
            # learn the representation of test case and patch. always use code2vec for test case.
            all_test_name, all_patch_name, all_test_vector, all_patch_vector, all_exception_type = self.test_patch_2vector(
                test_w2v=self.test_w2v, patch_w2v=self.patch_w2v)

            # save data with different types to pickle.
            datasets = [all_test_name, all_patch_name, all_test_vector, all_patch_vector, all_exception_type]
            pickle.dump(datasets, open(self.organized_dataset, 'wb'))

            self.test_name = datasets[0]
            self.patch_name = datasets[1]
            self.test_vector = datasets[2]
            self.patch_vector = datasets[3]
            self.exception_type = datasets[4]

    # make test/patch --->vector
    def test_patch_2vector(self, test_w2v, patch_w2v='cc2vec'):
        all_test_name, all_patch_name, all_test_vector, all_patch_vector, all_exception_type = [], [], [], [], []
        w2v = Word2vector(test_w2v=test_w2v, patch_w2v=patch_w2v, path_patch_root=self.path_patch_root)

        test_name_list = self.original_dataset[0]
        exception_type_list = self.original_dataset[1]
        log_list = self.original_dataset[2]
        test_function_list = self.original_dataset[3]
        # associated correct patch for failed test case. The name postfixing with '-one' means the complete patch hunk rather than part of it repaired the test case.
        patch_ids_list = self.original_dataset[4]
        for i in range(len(test_name_list)):
            # for i in tqdm(range(len(test_name_list))):
            name = test_name_list[i]
            function = test_function_list[i]
            ids = patch_ids_list[i]
            exception_type = exception_type_list[i]

            try:
                test_vector, patch_vector = w2v.convert_both(name, function, ids)
            except Exception as e:
                print('{} test name:{} exception emerge:{}'.format(i, name, e))
                continue
            print('{} test name:{} success!'.format(i, name, ))

            all_test_name.append(name)
            all_patch_name.append(ids)
            all_test_vector.append(test_vector)
            all_patch_vector.append(patch_vector)
            all_exception_type.append(exception_type)

        if self.patch_w2v == 'string':
            return all_test_name, all_patch_name, np.array(all_test_vector), all_patch_vector, all_exception_type
        else:
            return all_test_name, all_patch_name, np.array(all_test_vector), np.array(
                all_patch_vector), all_exception_type

    ####################################need to modify?
    def run(self, arg1='RQ2', arg2='', arg3='', arg4=''):
        # load original data and corresponding vector
        self.load_test()

        eval = evaluation(self.patch_w2v, self.test_w2v, self.original_dataset, self.test_name, self.test_vector,
                          self.patch_vector, self.exception_type,self.thre1,self.thre2)
        # RQ1: evaluate APOSTLE on the generated patches of APR tools.
        if 'apostle' == arg1:
            eval.predict_collected_projects(path_collected_patch=self.path_generated_patch, cut_off=0.8,
                                            distance_method=distance.cosine, ASE2020=False, patchsim=False, method=2,cof1=self.cof1,cof2=self.cof2 )

        # RQ2: compare ML-based approach.
        elif 'RQ2' == arg1:
            # ML-based approach.
            eval.predict_collected_projects(path_collected_patch=self.path_generated_patch, cut_off=0.8,
                                            distance_method=distance.cosine, ASE2020=True, patchsim=False, method=2, )
            # patchsim: run experiment/evaluate_patchsim.py
        elif 'method1' == arg1:
            eval.predict_collected_projects(path_collected_patch=self.path_generated_patch, cut_off=0.8,
                                            distance_method=distance.cosine, ASE2020=False, patchsim=False, method=1, )
        elif 'method2' == arg1:
            eval.predict_collected_projects(path_collected_patch=self.path_generated_patch, cut_off=0.8,
                                            distance_method=distance.cosine, ASE2020=False, patchsim=False, method=2, )
        elif 'method3' == arg1:
            eval.predict_collected_projects(path_collected_patch=self.path_generated_patch, cut_off=0.8,
                                            distance_method=distance.cosine, ASE2020=False, patchsim=False, method=3,)




if __name__ == '__main__':
    if len(sys.argv) == 2:
        script_name = sys.argv[0]
        arg1 = sys.argv[1]  # RQ
        arg2 = ''
        arg3 = ''
        arg4 = ''
    elif len(sys.argv) == 5:
        script_name = sys.argv[0]
        arg1 = sys.argv[1]  # predict
        arg2 = float(sys.argv[2])  # cut-off
        arg3 = sys.argv[3]  # project_id
        arg4 = sys.argv[4]  # path to patch snippet
    else:
        raise 'sorry, please check your arguments.'
    sys.argv = [sys.argv[0]]

    # specify RQ
    # arg1, arg2, arg3 = 'RQ2', '', ''

    config = Config()
    path_test = config.path_test
    path_patch_root = config.path_patch_root
    path_generated_patch = config.path_generated_patch
    organized_dataset = config.organized_dataset
    patch_w2v = config.patch_w2v
    test_w2v = config.test_w2v
    cof1=config.cof1
    cof2=config.cof2
    thre1=config.thre1
    thre2=config.thre2

    e = Experiment(path_test, path_patch_root, path_generated_patch, organized_dataset, patch_w2v, test_w2v,cof1,cof2,thre1,thre2)
    e.run(arg1, arg2, arg3, arg4)

