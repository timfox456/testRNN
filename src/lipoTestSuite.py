from keras.layers import *
from lipoClass import lipoClass
from SmilesEnumerator import SmilesEnumerator
from keras.preprocessing import image
from scipy import io
import itertools as iter
from testCaseGeneration import *
from testObjective import *
from oracle import *
from record import writeInfo
import random

def lipo_lstm_train():
    lipo = lipoClass()
    lipo.train_model()

def lipo_lstm_test(r,threshold_CC,threshold_MC,symbols_SQ,seq,TestCaseNum,minimalTest,TargMetri,CoverageStop):
    r.resetTime()
    random.seed(1)
    # set oracle radius
    oracleRadius = 0.2
    # load model
    lipo = lipoClass()
    lipo.load_data()
    lipo.load_model()

    # minimal test dataset generation
    if minimalTest != '0':
        ncdata = []
        ccdata = []
        mcdata = []
        sqpdata = []
        sqndata = []

    # test layer
    layer = 1
    termin = 0
    # predict logD value from smiles representation
    smiles = np.array(['CCC(=O)O[C@@]1(CC[NH+](C[C@H]1CC=C)C)c2ccccc2'])
    test = np.squeeze(lipo.smile_vect(smiles))
    h_t, c_t, f_t = lipo.cal_hidden_state(test)

    # input seeds
    X_train = lipo.X_train[random.sample(range(3100), 3000)]

    # test objective NC
    nctoe = NCTestObjectiveEvaluation(r)
    nctoe.model = lipo.model
    nctoe.testObjective.layer = layer
    nctoe.testCase = test
    activations_nc = nctoe.get_activations()
    nctoe.testObjective.feature = (np.argwhere(activations_nc >= np.min(activations_nc))).tolist()
    nctoe.testObjective.setOriginalNumOfFeature()

    # test objective CC
    cctoe = CCTestObjectiveEvaluation(r)
    cctoe.model = lipo.model
    cctoe.testObjective.layer = layer
    cctoe.hidden = h_t
    cctoe.threshold = float(threshold_CC)
    activations_cc = cctoe.get_activations()
    total_features_cc = (np.argwhere(activations_cc >= np.min(activations_cc))).tolist()
    cctoe.testObjective.feature = total_features_cc
    cctoe.testObjective.setOriginalNumOfFeature()
    cctoe.testObjective.setfeaturecount()

    # test objective MC
    mctoe = MCTestObjectiveEvaluation(r)
    mctoe.model = lipo.model
    mctoe.testObjective.layer = layer
    mctoe.hidden = f_t
    mctoe.threshold = float(threshold_MC)
    activations_mc = mctoe.get_activations()
    total_features_mc = (np.argwhere(activations_mc >= np.min(activations_mc))).tolist()
    mctoe.testObjective.feature = total_features_mc
    mctoe.testObjective.setOriginalNumOfFeature()
    mctoe.testObjective.setfeaturecount()

    # test objective SQ
    sqtoe = SQTestObjectiveEvaluation(r)
    sqtoe.model = lipo.model
    sqtoe.testObjective.layer = layer
    sqtoe.symbols = int(symbols_SQ)
    # generate all the features
    # choose time steps to cover
    t1 = int(seq[0])
    t2 = int(seq[1])
    indices = slice(t1, t2 + 1)
    #slice(70, 75)
    # characters to represent time series
    alpha_list = [chr(i) for i in range(97, 97 + int(symbols_SQ))]
    symb = ''.join(alpha_list)
    sqtoe.testObjective.feature_p = list(iter.product(symb, repeat=t2-t1+1))
    sqtoe.testObjective.feature_n = list(iter.product(symb, repeat=t2-t1+1))
    sqtoe.testObjective.setOriginalNumOfFeature()

    # define smile enumerator of a molecule
    sme = SmilesEnumerator()

    for test in X_train:
        for i in range(4):
            pred1 = lipo.displayInfo(test)
            # get next input test2 from the current input test
            smiles = lipo.vect_smile(np.array([test]))
            new_smiles = np.array([sme.randomize_smiles(smiles[0],i)])
            test2 = np.squeeze(lipo.smile_vect(new_smiles))

            if not (test2 is None):
                pred2 = lipo.displayInfo(test2)
                h_t, c_t, f_t = lipo.cal_hidden_state(test2)
                cctoe.hidden = h_t
                lipo.updateSample(pred1,pred2,0,True)
                # update NC coverage
                nctoe.testCase = test2
                nctoe.update_features()
                # update CC coverage
                cctoe.hidden = h_t
                cctoe.update_features()
                # update MC coverage
                mctoe.hidden = f_t
                mctoe.update_features()
                # update SQ coverage
                sqtoe.hidden = h_t
                sqtoe.update_features(indices)
                # write information to file
                writeInfo(r, lipo.numSamples, lipo.numAdv, lipo.perturbations, nctoe.coverage, cctoe.coverage, mctoe.coverage,
                          sqtoe.coverage_p, sqtoe.coverage_n)

                # terminate condition
                if TargMetri == 'CC':
                    termin = cctoe.coverage
                elif TargMetri == 'GC':
                    termin = mctoe.coverage
                elif TargMetri == 'SQN':
                    termin = sqtoe.coverage_n
                elif TargMetri == 'SQP':
                    termin = sqtoe.coverage_p


                # output test cases and adversarial example
                if minimalTest == '0':
                    f = open('output/smiles_test_set.txt', 'a')
                    f.write(new_smiles[0])
                    f.write('\n')
                    f.close()

                    if abs(pred1 - pred2) >= 1 :
                        f = open('adv_output/adv_smiles_test_set.txt', 'a')
                        f.write(new_smiles[0])
                        f.write('\n')
                        f.close()
                else:
                    if nctoe.minimal == 1 :
                        ncdata.append(test2)
                        f = open('minimal_nc/test_set.txt', 'a')
                        f.write(new_smiles[0])
                        f.write('\n')
                        f.close()
                    if cctoe.minimal == 1 :
                        ccdata.append(test2)
                        f = open('minimal_cc/test_set.txt', 'a')
                        f.write(new_smiles[0])
                        f.write('\n')
                        f.close()
                    if mctoe.minimal == 1 :
                        mcdata.append(test2)
                        f = open('minimal_mc/test_set.txt', 'a')
                        f.write(new_smiles[0])
                        f.write('\n')
                        f.close()
                    if sqtoe.minimalp == 1 :
                        sqpdata.append(test2)
                        f = open('minimal_sqp/test_set.txt', 'a')
                        f.write(new_smiles[0])
                        f.write('\n')
                        f.close()
                    if sqtoe.minimaln == 1 :
                        sqndata.append(test2)
                        f = open('minimal_sqn/test_set.txt', 'a')
                        f.write(new_smiles[0])
                        f.write('\n')
                        f.close()

            # check termination condition
            if lipo.numSamples < int(TestCaseNum) and termin < float(CoverageStop):
                continue
            else:
                io.savemat('log_folder/feature_count_CC.mat', {'feature_count_CC': cctoe.testObjective.feature_count})
                io.savemat('log_folder/feature_count_GC.mat', {'feature_count_GC': mctoe.testObjective.feature_count})
                # if minimalTest != '0':
                #     np.save('minimal_nc/ncdata', ncdata)
                #     np.save('minimal_cc/ccdata', ccdata)
                #     np.save('minimal_mc/mcdata', mcdata)
                #     np.save('minimal_sqp/sqpdata', sqpdata)
                #     np.save('minimal_sqn/sqndata', sqndata)
                break
        if lipo.numSamples < int(TestCaseNum) and termin < float(CoverageStop):
            continue
        else:
            break

    print("statistics: \n")
    nctoe.displayCoverage()
    cctoe.displayCoverage()
    mctoe.displayCoverage()
    sqtoe.displayCoverage1()
    sqtoe.displayCoverage2()
    lipo.displaySamples()
    lipo.displaySuccessRate()



