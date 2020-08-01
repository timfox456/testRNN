import argparse
import time
import sys
sys.path.append('example')
sys.path.append('src')
from utils import mkdir, delete_folder
from sentimentTestSuite import sentimentTrainModel, sentimentGenerateTestSuite
from mnistTestSuite import mnist_lstm_train, mnist_lstm_test
from lipoTestSuite import lipo_lstm_train, lipo_lstm_test
from tsTestSuite import ts_lstm_train, ts_lstm_test
from record import record
import re

def main():
    
    parser = argparse.ArgumentParser(description='testing for recurrent neural networks')
    parser.add_argument('--model', dest='modelName', default='lipo', help='')
    parser.add_argument('--TestCaseNum', dest='TestCaseNum', default='2000', help='')
    parser.add_argument('--TargMetri', dest='TargMetri', default='None', help='')
    parser.add_argument('--CoverageStop', dest='CoverageStop', default='0.9', help='')
    parser.add_argument('--threshold_CC', dest='threshold_CC', default='6', help='')
    parser.add_argument('--threshold_GC', dest='threshold_GC', default='0.78', help='')
    parser.add_argument('--symbols_SQ', dest='symbols_SQ', default='2', help='')
    parser.add_argument('--seq', dest='seq', default='[70,74]', help='')
    parser.add_argument('--mode', dest='mode', default='test', help='')
    parser.add_argument('--minimalTest', dest='minimalTest', default='0', help='')
    parser.add_argument('--output', dest='filename', default='./log_folder/record.txt', help='')
    
    args=parser.parse_args()
    
    modelName = args.modelName
    mode = args.mode
    filename = args.filename
    threshold_CC = args.threshold_CC
    threshold_MC = args.threshold_GC
    symbols_SQ = args.symbols_SQ
    seq = args.seq
    seq = re.findall(r"\d+\.?\d*", seq)
    TargMetri = args.TargMetri
    CoverageStop = args.CoverageStop
    TestCaseNum = args.TestCaseNum
    minimalTest = args.minimalTest
    # reset output folder
    if minimalTest == '0' :
        delete_folder("minimal_nc")
        delete_folder("minimal_cc")
        delete_folder("minimal_mc")
        delete_folder("minimal_sqp")
        delete_folder("minimal_sqn")
        mkdir("adv_output")
        mkdir("output")

    else:
        delete_folder("adv_output")
        delete_folder("output")
        mkdir("minimal_nc")
        mkdir("minimal_cc")
        mkdir("minimal_mc")
        mkdir("minimal_sqp")
        mkdir("minimal_sqn")

    # record time
    r = record(filename,time.time())
    if modelName == 'sentiment': 
        if mode == 'train': 
            sentimentTrainModel()
        else: 
            sentimentGenerateTestSuite(r,threshold_CC,threshold_MC,symbols_SQ,seq,TestCaseNum,minimalTest,TargMetri,CoverageStop)

    elif modelName == 'mnist':
        if mode == 'train':
            mnist_lstm_train()
        else:
            mnist_lstm_test(r,threshold_CC,threshold_MC,symbols_SQ,seq,TestCaseNum,minimalTest,TargMetri,CoverageStop)

    elif modelName == 'lipo':
        if mode == 'train':
            lipo_lstm_train()
        else:
            lipo_lstm_test(r,threshold_CC,threshold_MC,symbols_SQ,seq,TestCaseNum,minimalTest,TargMetri,CoverageStop)
        
    elif modelName == 'ts':
        if mode == 'train':
            ts_lstm_train()
        else:
            ts_lstm_test(r,threshold_CC,threshold_MC,symbols_SQ,seq,TestCaseNum,minimalTest,TargMetri,CoverageStop)

    else: 
        print("Please specify a model from {sentiment, mnist, lipo, ts}")
    
    r.close()

if __name__ == "__main__":

    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
