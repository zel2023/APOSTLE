import os

os.path.abspath(os.path.join('..', './representation'))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import pickle
#from representation.CC2Vec import lmg_cc2ftr_interface
from bert_serving.client import BertClient
# from gensim.models import word2vec, Doc2Vec
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import *
# Imports and method code2vec
from representation.code2vector import Code2vector
from representation.code2vec.config import Config
from representation.code2vec.model_base import Code2VecModelBase
import numpy as np
import re
import logging
from transformers import AutoTokenizer, AutoModel
from experiment.unixcoder import UniXcoder

# how to use
# tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
# model = AutoModel.from_pretrained("microsoft/codebert-base")
# nl_tokens=tokenizer.tokenize("return maximum value")
# ['return', 'Ġmaximum', 'Ġvalue']
# >>> code_tokens=tokenizer.tokenize("def max(a,b): if a>b: return a else return b")
# ['def', 'Ġmax', '(', 'a', ',', 'b', '):', 'Ġif', 'Ġa', '>', 'b', ':', 'Ġreturn', 'Ġa', 'Ġelse', 'Ġreturn', 'Ġb']
# >>> tokens=[tokenizer.cls_token]+nl_tokens+[tokenizer.sep_token]+code_tokens+[tokenizer.eos_token]
# ['<s>', 'return', 'Ġmaximum', 'Ġvalue', '</s>', 'def', 'Ġmax', '(', 'a', ',', 'b', '):', 'Ġif', 'Ġa', '>', 'b', ':', 'Ġreturn', 'Ġa', 'Ġelse', 'Ġreturn', 'Ġb', '</s>']
# >>> tokens_ids=tokenizer.convert_tokens_to_ids(tokens)
# [0, 30921, 4532, 923, 2, 9232, 19220, 1640, 102, 6, 428, 3256, 114, 10, 15698, 428, 35, 671, 10, 1493, 671, 741, 2]
# >>> context_embeddings=model(torch.tensor(tokens_ids)[None,:])[0]
# torch.Size([1, 23, 768])
# tensor([[-0.1423,  0.3766,  0.0443,  ..., -0.2513, -0.3099,  0.3183],
#         [-0.5739,  0.1333,  0.2314,  ..., -0.1240, -0.1219,  0.2033],
#         [-0.1579,  0.1335,  0.0291,  ...,  0.2340, -0.8801,  0.6216],
#         ...,
#         [-0.4042,  0.2284,  0.5241,  ..., -0.2046, -0.2419,  0.7031],
#         [-0.3894,  0.4603,  0.4797,  ..., -0.3335, -0.6049,  0.4730],
#         [-0.1433,  0.3785,  0.0450,  ..., -0.2527, -0.3121,  0.3207]],
#        grad_fn=<SelectBackward>)

MODEL_MODEL_LOAD_PATH = '../models/java14_model/saved_model_iter8.release'
MODEL_CC2Vec = '../representation/CC2Vec/'
MODEL_CODEBERT="../representation/codebert/"

class Word2vector:
    def __init__(self, test_w2v=None, patch_w2v=None,thre1=None,thre2=None, path_patch_root=None):
        # self.w2v = word2vec
        self.test_w2v = test_w2v
        self.patch_w2v = patch_w2v
        self.path_patch_root = path_patch_root
        self.error = 0
        self.error2 = 0
        self.thre1 = thre1
        self.thre2 = thre2
        self.patch = ""
        # Init and Load the model for test cases
        if self.test_w2v == 'code2vec':
            config = Config(set_defaults=True, load_from_args=True, verify=False)
            config.MODEL_LOAD_PATH = MODEL_MODEL_LOAD_PATH
            config.EXPORT_CODE_VECTORS = True
            model = Word2vector.load_model_code2vec_dynamically(config)
            config.log('Done creating code2vec model')
            self.c2v = Code2vector(self.test_w2v, model, 'none', 'none')
        elif self.test_w2v == 'codebert':
            #self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_CODEBERT)
            #self.model = AutoModel.from_pretrained("microsoft/codebert-base")
            self.model = AutoModel.from_pretrained(MODEL_CODEBERT)
            config = Config(set_defaults=True, load_from_args=True, verify=False)
            config.EXPORT_CODE_VECTORS = True
            config.log('------------Done creating codebert model---------------')
            self.c2v = Code2vector(self.test_w2v, self.model, self.tokenizer, 'none')
        elif self.test_w2v == 'unixcoder':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = UniXcoder("microsoft/unixcoder-base")
            self.model.to(self.device)
            config = Config(set_defaults=True, load_from_args=True, verify=False)
            config.EXPORT_CODE_VECTORS = True
            config.log('------------Done creating unixcoder model---------------')
            self.c2v = Code2vector(self.test_w2v, self.model, 'none', self.device)
            # =======================
        elif self.test_w2v == 'graphcodebert':
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
            self.model = AutoModel.from_pretrained("microsoft/graphcodebert-base")
            config = Config(set_defaults=True, load_from_args=True, verify=False)
            config.EXPORT_CODE_VECTORS = True
            config.log('------------Done creating graphcodebert model---------------')
            self.c2v = Code2vector(self.test_w2v, self.model, self.tokenizer, 'none')
        # init for patch vector
        if self.patch_w2v == 'cc2vec':
            self.dictionary = pickle.load(open(MODEL_CC2Vec + 'dict.pkl', 'rb'))
        elif self.patch_w2v == 'bert':
            import ssl
            try:
                _create_unverified_https_context = ssl._create_unverified_context
            except AttributeError:
                pass
            else:
                ssl._create_default_https_context = _create_unverified_https_context
            # nltk.download('punkt')

            logging.getLogger().info('Waiting for Bert server')
            self.m = BertClient(check_length=False, check_version=False)
        elif self.patch_w2v == 'codebert':
            #self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
            #self.model = AutoModel.from_pretrained("microsoft/codebert-base")
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_CODEBERT)
            self.model = AutoModel.from_pretrained(MODEL_CODEBERT)


            ########
        elif self.patch_w2v == 'unixcoder':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = UniXcoder("microsoft/unixcoder-base")
            self.model.to(self.device)
        elif self.patch_w2v == 'graphcodebert':
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
            self.model = AutoModel.from_pretrained("microsoft/graphcodebert-base")

    @staticmethod
    def load_model_code2vec_dynamically(config: Config) -> Code2VecModelBase:
        assert config.DL_FRAMEWORK in {'tensorflow', 'keras'}
        if config.DL_FRAMEWORK == 'tensorflow':
            from representation.code2vec.tensorflow_model import Code2VecModel
        elif config.DL_FRAMEWORK == 'keras':
            from representation.code2vec.keras_model import Code2VecModel
        return Code2VecModel(config)

    # 将测试用例和补丁向量化
    def convert_both(self, test_name, test_text, patch_ids):
        try:
            # 1 represent test cases by embedding test function with code2vec(output dimension: 384)
            function = test_text
            test_vector = self.c2v.convert(function)

            # 2 represent patch
            # first of all, find associated patch path
            if len(patch_ids) == 1 and patch_ids[0].endswith('-one'):
                # '-one' means this complete patch is the solution for the failed test case
                project = patch_ids[0].split('_')[0]
                id = patch_ids[0].split('_')[1].replace('-one', '')
                path_patch = self.path_patch_root + project + '/' + id + '/'
                patch_ids = os.listdir(path_patch)
                path_patch_ids = [path_patch + patch_id for patch_id in patch_ids]
            else:
                path_patch_ids = []
                for name in patch_ids:
                    project = name.split('_')[0]
                    id = name.split('_')[1]
                    patch_id = name.split('_')[1] + '_' + name.split('_')[2] + '.patch'
                    path_patch = self.path_patch_root + project + '/' + id + '/'
                    path_patch_ids.append(os.path.join(path_patch, patch_id))

            # embedding each patch and combine them together to learn the overall behaviour
            multi_vector = []
            for path_patch_id in path_patch_ids:
                if self.patch_w2v == 'cc2vec':
                    learned_vector = lmg_cc2ftr_interface.learned_feature(path_patch_id,
                                                                          load_model=MODEL_CC2Vec + 'cc2ftr.pt',
                                                                          dictionary=self.dictionary)
                    learned_vector = list(learned_vector.flatten())
                elif self.patch_w2v == 'bert':
                    learned_vector = self.learned_feature(path_patch_id, self.patch_w2v)
                elif self.patch_w2v == 'string':
                    learned_vector = self.extract_text(path_patch_id, )
                elif self.patch_w2v == 'codebert':
                    learned_vector = self.learned_feature(path_patch_id, self.patch_w2v)
                    # codebert patch embedding over
                elif self.patch_w2v == 'unixcoder':
                    learned_vector = self.learned_feature(path_patch_id, self.patch_w2v)
                elif self.patch_w2v == 'graphcodebert':
                    learned_vector = self.learned_feature(path_patch_id, self.patch_w2v)
                multi_vector.append(learned_vector)

            # if self.patch_w2v == 'string':
            #     patch_vector = ''
            #     for s in multi_vector:
            #         patch_vector += s
            #     patch_vector = [patch_vector]
            else:
                patch_vector = np.array(multi_vector).sum(axis=0)
        except Exception as e:
            raise e

        if self.patch_w2v == 'string':
            if patch_vector == ['']:
                raise Exception('null patch string')
            return test_vector, patch_vector
        else:
            if test_vector.size == 0 or patch_vector.size == 0:
                raise Exception('null vector')
            return test_vector, patch_vector

    # def convert(self, test_name, data_text):
    #     if self.test_w2v == 'code2vec':
    #         test_vector = []
    #         for i in range(len(data_text)):
    #             function = data_text[i]
    #             try:
    #                 vector = self.c2v.convert(function)
    #             except Exception as e:
    #                 print('{} test_name:{} Exception:{}'.format(i, test_name[i], 'Wrong syntax'))
    #                 continue
    #             print('{} test_name:{}'.format(i, test_name[i]))
    #             test_vector.append(vector)
    #         return test_vector
    #
    #     if self.patch_w2v == 'cc2vec':
    #         patch_vector = []
    #         for i in range(len(data_text)):
    #             patch_ids = data_text[i]
    #             # find path_patch
    #             if len(patch_ids) == 1 and patch_ids[0].endwith('-one'):
    #                 project = patch_ids[0].split('_')[0]
    #                 id = patch_ids[0].split('_')[1].replace('-one','')
    #                 path_patch = self.path_patch_root + project +'/'+ id + '/'
    #                 patch_ids = os.listdir(path_patch)
    #                 path_patch_ids = [path_patch + patch_id for patch_id in patch_ids]
    #             else:
    #                 path_patch_ids = []
    #                 for name in patch_ids:
    #                     project = name.split('_')[0]
    #                     id = name.split('_')[1]
    #                     patch_id = name.split('_')[1] +'_'+ name.split('_')[2] + '.patch'
    #                     path_patch = self.path_patch_root + project +'/'+ id + '/'
    #                     path_patch_ids.append(os.path.join(path_patch, patch_id))
    #
    #             multi_vector = []
    #             for path_patch_id in path_patch_ids:
    #                 learned_vector = lmg_cc2ftr_interface.learned_feature(path_patch_id, load_model=MODEL_CC2Vec+'cc2ftr.pt', dictionary=self.dictionary)
    #                 multi_vector.append(list(learned_vector.flatten()))
    #             combined_vector = np.array(multi_vector).mean(axis=0)
    #             patch_vector.append(combined_vector)
    #         return patch_vector


    def convert_single_patch(self, path_patch):
        try:
            if self.patch_w2v == 'cc2vec':
                self.error = 0
                self.error2 = 0
                multi_vector = []  # sum up vectors of different parts of patch
                # patch = os.listdir(path_patch)
                # for part in patch:
                for root, dirs, files in os.walk(path_patch):
                    for file in files:
                        if file.endswith('.patch'):  # 把补丁文件夹下的补丁文件拿出来
                            p = os.path.join(root, file)  # p的值就是具体的补丁文件路径
                            # print("p:")
                            # print(p);
                            learned_vector = lmg_cc2ftr_interface.learned_feature(p,
                                                                                  load_model=MODEL_CC2Vec + 'cc2ftr.pt',
                                                                                  dictionary=self.dictionary)

                            multi_vector.append(list(learned_vector.flatten()))
                combined_vector = np.array(multi_vector).sum(axis=0)

            elif self.patch_w2v == 'bert':
                multi_vector = []
                multi_vector_cross = []
                patch = os.listdir(path_patch)
                for part in patch:
                    p = os.path.join(path_patch, part)
                    learned_vector = self.learned_feature(p, self.patch_w2v)
                    learned_vector_cross = self.learned_feature_cross(p, self.patch_w2v)

                    multi_vector.append(learned_vector)
                    multi_vector_cross.append(learned_vector_cross)
                combined_vector = np.array(multi_vector).sum(axis=0)
                combined_vector_cross = np.array(multi_vector_cross).sum(axis=0)
                return combined_vector, combined_vector_cross

            elif self.patch_w2v == 'codebert':
                self.error = 0
                self.error2 = 0
                multi_vector = []  # sum up vectors of different parts of patch
                # patch = os.listdir(path_patch)
                # for part in patch:
                self.error = 0
                self.error2 = 0
                # self.patch=""
                for root, dirs, files in os.walk(path_patch):
                    for file in files:
                        if file.endswith('.patch'):  # 把补丁文件夹下的补丁文件拿出来
                            p = os.path.join(root, file)  # p的值就是具体的补丁文件路径
                            # print("p:")
                            # print(p);
                            learned_vector = self.learned_feature(p, self.patch_w2v)
                            # print("learned_vector:")
                            # print(learned_vector);
                            multi_vector.append(list(learned_vector.flatten()))
                # if self.error>1500:
                #    self.error=1
                # else:
                #    self.error=0
                combined_vector = np.array(multi_vector).sum(axis=0)
            elif self.patch_w2v == 'unixcoder':
                self.error = 0
                self.error2 = 0
                multi_vector = []  # sum up vectors of different parts of patch
                # patch = os.listdir(path_patch)
                # for part in patch:
                self.error = 0
                for root, dirs, files in os.walk(path_patch):
                    for file in files:
                        if file.endswith('.patch'):  # 把补丁文件夹下的补丁文件拿出来
                            p = os.path.join(root, file)  # p的值就是具体的补丁文件路径
                            # print("p:")
                            # print(p);
                            learned_vector = self.learned_feature(p, self.patch_w2v)
                            # print("learned_vector:")
                            # print(learned_vector);
                            multi_vector.append(list(learned_vector.flatten()))
                combined_vector = np.array(multi_vector).sum(axis=0)
            elif self.patch_w2v == 'graphcodebert':
                self.error = 0
                self.error2 = 0
                multi_vector = []  # sum up vectors of different parts of patch
                # patch = os.listdir(path_patch)
                # for part in patch:
                for root, dirs, files in os.walk(path_patch):
                    for file in files:
                        if file.endswith('.patch'):  # 把补丁文件夹下的补丁文件拿出来
                            p = os.path.join(root, file)  # p的值就是具体的补丁文件路径
                            # print("p:")
                            # print(p);
                            learned_vector = self.learned_feature(p, self.patch_w2v)
                            # print("learned_vector:")
                            # print(learned_vector);
                            multi_vector.append(list(learned_vector.flatten()))
                combined_vector = np.array(multi_vector).sum(axis=0)
            return combined_vector, self.error, self.error2, None
        except Exception as e:
            raise e

    def extract_text(self, path_patch, ):
        try:
            bugy_all = self.get_only_change(path_patch, type='buggy')
            patched_all = self.get_only_change(path_patch, type='patched')
        except Exception as e:
            # print('patch: {}, exception: {}'.format(path_patch, e))
            raise e
        return bugy_all + patched_all

    def learned_feature(self, path_patch, w2v):
        if w2v == 'codebert':
            bugy_all = self.get_only_change(path_patch, type='buggy')
            patched_all = self.get_only_change(path_patch, type='patched')
            # self.error=(self.error,abs(len(bugy_all)+len(patched_all)))
            # self.error+=len(bugy_all)+len(patched_all)
            if (len(bugy_all) + len(patched_all) > self.thre1):
                self.error2 = 1
            if (len(bugy_all) > 1023):
                bugy_all = bugy_all[0:1023]
            if (len(patched_all) > 1023):
                patched_all = patched_all[0:1023]
            # nl_tokens = self.tokenizer.tokenize("")
            code_tokens = self.tokenizer.tokenize(bugy_all)
            # tokens = [self.tokenizer.cls_token] + nl_tokens + [self.tokenizer.sep_token] + code_tokens + [
            #    self.tokenizer.eos_token]
            tokens = [self.tokenizer.cls_token] + code_tokens + [self.tokenizer.eos_token]
            tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            vector = self.model(torch.tensor(tokens_ids)[None, :])[0]
            new_test_vec = []
            for ts in vector:
                ts = ts.detach().numpy()
                ts = np.mean(ts, axis=0)
                new_test_vec.append(ts)
            bug_vec = np.mean(new_test_vec, axis=0)

            code_tokens = self.tokenizer.tokenize(patched_all)

            tokens = [self.tokenizer.cls_token] + code_tokens + [self.tokenizer.eos_token]
            tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            vector = self.model(torch.tensor(tokens_ids)[None, :])[0]
            new_test_vec = []
            for ts in vector:
                ts = ts.detach().numpy()
                ts = np.mean(ts, axis=0)
                new_test_vec.append(ts)
            patched_vec = np.mean(new_test_vec, axis=0)

            dist = distance.euclidean(patched_vec, bug_vec) / (1 + distance.euclidean(patched_vec, bug_vec))
            if dist > self.thre2:  # we use similarity instead of distance
                self.error = 1
            embedding = self.subtraction(bug_vec, patched_vec);

            return embedding
        elif w2v == 'unixcoder':
            bugy_all = self.get_only_change(path_patch, type='buggy')
            patched_all = self.get_only_change(path_patch, type='patched')

            if (len(bugy_all) + len(patched_all) > self.thre1):
                self.error = 1

            tokens_ids = self.model.tokenize([bugy_all], max_length=512, mode="<encoder-only>")
            source_ids = torch.tensor(tokens_ids).to(self.device)
            tokens_embeddings, func_embedding = self.model(source_ids)

            new_test_vec = []
            for ts in func_embedding:
                ts = ts.detach().numpy()
                # ts=np.mean(ts, axis=0)
                new_test_vec.append(ts)
            bug_vec = np.mean(new_test_vec, axis=0)

            tokens_ids = self.model.tokenize([patched_all], max_length=512, mode="<encoder-only>")
            source_ids = torch.tensor(tokens_ids).to(self.device)
            tokens_embeddings, func_embedding = self.model(source_ids)

            new_test_vec = []
            for ts in func_embedding:
                # torch.squeeze(ts)
                ts = ts.detach().numpy()
                # ts=np.mean(ts, axis=0)
                new_test_vec.append(ts)
            patched_vec = np.mean(new_test_vec, axis=0)

            dist = distance.euclidean(patched_vec, bug_vec) / (1 + distance.euclidean(patched_vec, bug_vec))
            if dist > self.thre2:  # we use similarity instead of distance
                self.error = 1
            embedding = self.subtraction(bug_vec, patched_vec)

            return embedding
        elif w2v == 'graphcodebert':
            bugy_all = self.get_only_change(path_patch, type='buggy')
            patched_all = self.get_only_change(path_patch, type='patched')

            if (len(bugy_all) + len(patched_all) > self.thre1):
                self.error2 = 1

            if (len(bugy_all) > 1023):
                bugy_all = bugy_all[0:1023]
            if (len(patched_all) > 1023):
                patched_all = patched_all[0:1023]
            # nl_tokens = self.tokenizer.tokenize("")
            code_tokens = self.tokenizer.tokenize(bugy_all)
            # tokens = [self.tokenizer.cls_token] + nl_tokens + [self.tokenizer.sep_token] + code_tokens + [
            #    self.tokenizer.eos_token]
            tokens = [self.tokenizer.cls_token] + code_tokens + [self.tokenizer.sep_token]
            tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            vector = self.model(torch.tensor(tokens_ids)[None, :])[0][0, 0]
            # self.vector=np.array(self.vector)
            bug_vec = vector.detach().numpy()

            code_tokens = self.tokenizer.tokenize(patched_all)
            # tokens = [self.tokenizer.cls_token] + nl_tokens + [self.tokenizer.sep_token] + code_tokens + [
            #    self.tokenizer.eos_token]
            tokens = [self.tokenizer.cls_token] + code_tokens + [self.tokenizer.sep_token]
            tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            vector = self.model(torch.tensor(tokens_ids)[None, :])[0][0, 0]
            patched_vec = vector.detach().numpy()

            dist = distance.euclidean(patched_vec, bug_vec) / (1 + distance.euclidean(patched_vec, bug_vec))
            if dist > self.thre2:  # we use similarity instead of distance
                self.error = 1

            embedding = self.subtraction(bug_vec, patched_vec)
            return embedding
        else:
            try:
                # bugy_all = self.get_diff_files_frag(path_patch, type='buggy')
                # patched_all = self.get_diff_files_frag(path_patch, type='patched')
                bugy_all = self.get_only_change(path_patch, type='buggy')
                patched_all = self.get_only_change(path_patch, type='patched')

                # tokenize word
                bugy_all_token = word_tokenize(bugy_all)
                patched_all_token = word_tokenize(patched_all)

                bug_vec, patched_vec = self.output_vec(w2v, bugy_all_token, patched_all_token)
            except Exception as e:
                # print('patch: {}, exception: {}'.format(path_patch, e))
                raise e

            bug_vec = bug_vec.reshape((1, -1))
            patched_vec = patched_vec.reshape((1, -1))

            # embedding feature cross
            # subtract, multiple, cos, euc = self.multi_diff_features(bug_vec, patched_vec)
            # embedding = np.hstack((subtract, multiple, cos, euc,))

            embedding = self.subtraction(bug_vec, patched_vec)

            return list(embedding.flatten())

    def learned_feature_cross(self, path_patch, w2v):
        try:
            bugy_all = self.get_diff_files_frag(path_patch, type='buggy')
            patched_all = self.get_diff_files_frag(path_patch, type='patched')

            # tokenize word
            bugy_all_token = word_tokenize(bugy_all)
            patched_all_token = word_tokenize(patched_all)

            bug_vec, patched_vec = self.output_vec(w2v, bugy_all_token, patched_all_token)
        except Exception as e:
            # print('patch: {}, exception: {}'.format(path_patch, e))
            raise e

        bug_vec = bug_vec.reshape((1, -1))
        patched_vec = patched_vec.reshape((1, -1))

        # embedding feature cross
        subtract, multiple, cos, euc = self.multi_diff_features(bug_vec, patched_vec)
        embedding = np.hstack((subtract, multiple, cos, euc,))

        return list(embedding.flatten())

    def subtraction(self, buggy, patched):
        return buggy - patched

    def multiplication(self, buggy, patched):
        return buggy * patched

    def cosine_similarity(self, buggy, patched):
        return paired_cosine_distances(buggy, patched)

    def euclidean_similarity(self, buggy, patched):
        return paired_euclidean_distances(buggy, patched)

    def multi_diff_features(self, buggy, patched):
        subtract = self.subtraction(buggy, patched)
        multiple = self.multiplication(buggy, patched)
        cos = self.cosine_similarity(buggy, patched).reshape((1, 1))
        euc = self.euclidean_similarity(buggy, patched).reshape((1, 1))

        return subtract, multiple, cos, euc

    def output_vec(self, w2v, bugy_all_token, patched_all_token):

        if w2v == 'bert':
            if bugy_all_token == []:
                bug_vec = np.zeros((1, 1024))
            else:
                bug_vec = self.m.encode([bugy_all_token], is_tokenized=True)

            if patched_all_token == []:
                patched_vec = np.zeros((1, 1024))
            else:
                patched_vec = self.m.encode([patched_all_token], is_tokenized=True)
        elif w2v == 'doc':
            # m = Doc2Vec.load('../model/doc_file_64d.model')
            m = Doc2Vec.load('../model/Doc_frag_ASE.model')
            bug_vec = m.infer_vector(bugy_all_token, alpha=0.025, steps=300)
            patched_vec = m.infer_vector(patched_all_token, alpha=0.025, steps=300)
        else:
            print('wrong model')
            raise

        return bug_vec, patched_vec

    def get_only_change(self, path_patch, type='patched'):
        with open(path_patch, 'r+') as file:
            lines = ''
            p = r"([^\w_])"
            # try:
            for line in file:
                line = line.strip()
                if line != '':
                    if line.startswith('@@') or line.startswith('diff') or line.startswith('index'):
                        continue
                    elif type == 'buggy':
                        if line.startswith('--- ') or line.startswith('-- ') or line.startswith('PATCH_DIFF_ORIG=---'):
                            continue
                        elif line.startswith('-'):
                            if line[1:].strip() == '':
                                continue
                            if line[1:].strip().startswith('//'):
                                continue
                            line = re.split(pattern=p, string=line[1:].strip())

                            final = []
                            for s in line:
                                s = s.strip()
                                if s == '' or s == ' ':
                                    continue
                                # CamelCase to underscore_case
                                cases = re.split(pattern='(?=[A-Z0-9])', string=s)
                                for c in cases:
                                    if c == '' or c == ' ':
                                        continue
                                    final.append(c)

                            line = ' '.join(final)
                            lines += line.strip() + ' '
                        else:
                            # do nothing
                            pass
                    elif type == 'patched':
                        if line.startswith('+++ ') or line.startswith('++ '):
                            continue
                            # line = re.split(pattern=p, string=line.split(' ')[1].strip())
                            # lines += ' '.join(line) + ' '
                        elif line.startswith('+'):
                            if line[1:].strip() == '':
                                continue
                            if line[1:].strip().startswith('//'):
                                continue
                            line = re.split(pattern=p, string=line[1:].strip())
                            final = []
                            for s in line:
                                s = s.strip()
                                if s == '' or s == ' ':
                                    continue
                                # CamelCase to underscore_case
                                cases = re.split(pattern='(?=[A-Z0-9])', string=s)
                                for c in cases:
                                    if c == '' or c == ' ':
                                        continue
                                    final.append(c)
                            line = ' '.join(final)
                            lines += line.strip() + ' '
                        else:
                            # do nothing
                            pass
        return lines

    def get_diff_files_frag(self, path_patch, type):
        with open(path_patch, 'r') as file:
            lines = ''
            p = r"([^\w_])"
            flag = True
            # try:
            for line in file:
                line = line.strip()
                if '*/' in line:
                    flag = True
                    continue
                if flag == False:
                    continue
                if line != '':
                    if line.startswith('@@') or line.startswith('diff') or line.startswith('index'):
                        continue
                    if line.startswith('Index') or line.startswith('==='):
                        continue
                    elif '/*' in line:
                        flag = False
                        continue
                    elif type == 'buggy':
                        if line.startswith('--- ') or line.startswith('-- ') or line.startswith('PATCH_DIFF_ORIG=---'):
                            continue
                            # line = re.split(pattern=p, string=line.split(' ')[1].strip())
                            # lines += ' '.join(line) + ' '
                        elif line.startswith('-'):
                            if line[1:].strip() == '':
                                continue
                            if line[1:].strip().startswith('//'):
                                continue
                            line = re.split(pattern=p, string=line[1:].strip())
                            final = []
                            for s in line:
                                s = s.strip()
                                if s == '' or s == ' ':
                                    continue
                                # CamelCase to underscore_case
                                cases = re.split(pattern='(?=[A-Z0-9])', string=s)
                                for c in cases:
                                    if c == '' or c == ' ':
                                        continue
                                    final.append(c)
                            line = ' '.join(final)
                            lines += line.strip() + ' '
                        elif line.startswith('+'):
                            # do nothing
                            pass
                        else:
                            line = re.split(pattern=p, string=line.strip())
                            final = []
                            for s in line:
                                s = s.strip()
                                if s == '' or s == ' ':
                                    continue
                                # CamelCase to underscore_case
                                cases = re.split(pattern='(?=[A-Z0-9])', string=s)
                                for c in cases:
                                    if c == '' or c == ' ':
                                        continue
                                    final.append(c)
                            line = ' '.join(final)
                            lines += line.strip() + ' '
                    elif type == 'patched':
                        if line.startswith('+++ ') or line.startswith('++ '):
                            continue
                            # line = re.split(pattern=p, string=line.split(' ')[1].strip())
                            # lines += ' '.join(line) + ' '
                        elif line.startswith('+'):
                            if line[1:].strip() == '':
                                continue
                            if line[1:].strip().startswith('//'):
                                continue
                            line = re.split(pattern=p, string=line[1:].strip())
                            final = []
                            for s in line:
                                s = s.strip()
                                if s == '' or s == ' ':
                                    continue
                                # CamelCase to underscore_case
                                cases = re.split(pattern='(?=[A-Z0-9])', string=s)
                                for c in cases:
                                    if c == '' or c == ' ':
                                        continue
                                    final.append(c)
                            line = ' '.join(final)
                            lines += line.strip() + ' '
                        elif line.startswith('-'):
                            # do nothing
                            pass
                        else:
                            line = re.split(pattern=p, string=line.strip())
                            final = []
                            for s in line:
                                s = s.strip()
                                if s == '' or s == ' ':
                                    continue
                                # CamelCase to underscore_case
                                cases = re.split(pattern='(?=[A-Z0-9])', string=s)
                                for c in cases:
                                    if c == '' or c == ' ':
                                        continue
                                    final.append(c)
                            line = ' '.join(final)
                            lines += line.strip() + ' '
            # except Exception:
            #     print(Exception)
            #     return 'Error'
            return lines


