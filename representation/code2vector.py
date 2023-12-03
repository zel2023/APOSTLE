from representation.code2vec.extractor import Extractor
import tempfile
import torch
import numpy as np

EXTRACTOR_JAR = "../representation/code2vec/JavaExtractor/JPredict/target/JavaExtractor-0.0.1-SNAPSHOT.jar"
MAX_CONTEXTS = 200
MAX_PATH_LENGTH = 8
MAX_PATH_WIDTH = 2

class Code2vector:
    def __init__(self,test_w2v, model,tokenizer,device):
        self.model = model
        self.vector = None
        if test_w2v=='codebert':
            self.tokenizer=tokenizer
        if test_w2v=='graphcodebert':
            self.tokenizer=tokenizer
        if test_w2v=='unixcoder':
            self.device=device
        self.test_w2v=test_w2v

    def convert(self, function):
        if self.test_w2v == 'code2vec':
            f = tempfile.NamedTemporaryFile(mode='w+', dir='../tmp', delete=True)
            f.write(function)
            file_path = f.name
            f.seek(0)
            extractor = Extractor(MAX_CONTEXTS, EXTRACTOR_JAR, MAX_PATH_LENGTH, MAX_PATH_WIDTH)
            paths, _ = extractor.extract_paths(file_path)
            f.close()
            result = self.model.predict(paths)

            if result:
                self.vector = result[0].code_vector
            return self.vector
        elif self.test_w2v == 'codebert':
            if((len(function))>511):
                function=function[0:511]
            code_tokens = self.tokenizer.tokenize(function)
            tokens = [self.tokenizer.cls_token]  + code_tokens + [  self.tokenizer.eos_token]
            tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            self.vector = self.model(torch.tensor(tokens_ids)[None, :])[0]
            new_test_vec=[]
            for ts in self.vector:
                #torch.squeeze(ts)
                ts= ts.detach().numpy()
                ts=np.mean(ts, axis=0)
                new_test_vec.append(ts)
            new_test_vec=np.mean(new_test_vec,axis=0)
            return new_test_vec
        elif self.test_w2v == 'unixcoder':
            # Encode  function
            
            tokens_ids = self.model.tokenize([function], max_length=1023, mode="<encoder-only>")
            source_ids = torch.tensor(tokens_ids).to(self.device)
            tokens_embeddings, func_embedding = self.model(source_ids)
            #print(tokens_embeddings)
            #print(torch.Size())
            #print("2")
            #print(max_func_embedding)
            #return tokens_embeddings
            new_test_vec=[]
            for ts in func_embedding:
                #torch.squeeze(ts)
                ts= ts.detach().numpy()
                #ts=np.mean(ts, axis=0)
                new_test_vec.append(ts)
            new_test_vec=np.mean(new_test_vec,axis=0)
            #new_test_vec=new_test_vec.reshape((1, -1))
            #print(len(new_test_vec))
            return new_test_vec
        elif self.test_w2v == 'graphcodebert':
            # if((len(function))>511):
            #     function=function[0:511]
            code_tokens = self.tokenizer.tokenize("def max(a,b): if a>b: return a else return b")
            tokens = [self.tokenizer.cls_token] + code_tokens + [self.tokenizer.sep_token]
            tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            self.vector = self.model(torch.tensor(tokens_ids)[None, :])[0][0, 0]
            #self.vector=np.array(self.vector)
            new_test_vec=self.vector.detach().numpy()
            #print(new_test_vec)
            # code_tokens = self.tokenizer.tokenize(function)
            # tokens = [self.tokenizer.cls_token]  + code_tokens + [  self.tokenizer.eos_token]
            # tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            # self.vector = self.model(torch.tensor(tokens_ids)[None, :])[0]
            return new_test_vec

