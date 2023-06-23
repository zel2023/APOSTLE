class Config:
    def __init__(self):
        # the orginal data of test case name, test function, associated patch including 'single test case' and 'full' versions.
        self.path_test = '../data/test_case_all.pkl'
        # developers' patches in defects4j
        self.path_patch_root = '../data/defects4j_patch_sliced/'

        #  generated patches of APR tools
        self.path_generated_patch = '../data/APOSTLE_DataSet/PatchCollectingV1_sliced/'
        # for Naturalness

        #choose the test_w2v to learn the behaviour of test cases
        #self.test_w2v='codebert'
        self.test_w2v='unixcoder'
        #self.test_w2v='code2vec'
        #self.test_w2v='graphcodebert'

        # choose one type of representations to learn the behaviour of patch
        self.patch_w2v='codebert'
        #self.patch_w2v = 'unixcoder'
        #self.patch_w2v='graphcodebert'

        self.organized_dataset = '../data/organized_dataset_' + self.patch_w2v+'_'+self.test_w2v +'.pickle'
        
        self.cof1=0.55
        self.cof2=0.95
        self.thre1=700
        self.thre2=0.2


if __name__ == '__main__':
    Config()
