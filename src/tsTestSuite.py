from keras.layers import *
from tsClass import tsClass
from SmilesEnumerator import SmilesEnumerator
from keras.preprocessing import image
from scipy import io
import itertools as iter
from testCaseGeneration import *
from testObjective import *
from oracle import *
from record import writeInfo
import random

def ts_lstm_train():
    ts = tsClass()
    ts.train_model()

def ts_lstm_test(r,threshold_CC,threshold_MC,symbols_SQ,seq,TestCaseNum,minimalTest,TargMetri,CoverageStop):
    r.resetTime()
    random.seed(1)
    # set oracle radius
    oracleRadius = 0.2
    # load model
    ts = tsClass()
    ts.load_data('dataset/sp500.csv', 50, True)
    model = ts.load_model()
    if not model:
        ts.train_model()


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

    test_data = [
        1455.219971,
        1399.420044,
        1402.109985,
        1403.449951,
        1441.469971,
        1457.599976,
        1438.560059,
        1432.25,
        1449.680054,
        1465.150024,
        1455.140015,
        1455.900024,
        1445.569946,
        1441.359985,
        1401.530029,
        1410.030029,
        1404.089966,
        1398.560059,
        1360.160034,
        1394.459961,
        1409.280029,
        1409.119995,
        1424.969971,
        1424.369995,
        1424.23999,
        1441.719971,
        1411.709961,
        1416.829956,
        1387.119995,
        1389.939941,
        1402.050049,
        1387.670044,
        1388.26001,
        1346.089966,
        1352.170044,
        1360.689941,
        1353.430054,
        1333.359985,
        1348.050049,
        1366.420044,
        1379.189941,
        1381.76001,
        1409.170044,
        1391.280029,
        1355.619995,
        1366.699951,
        1401.689941,
        1395.069946,
        1383.619995,
        1359.150024
    ]
    # predict logD value from smiles representation
    test_array = np.array(test_data).reshape((1,50,1))
    test = np.array(ts.create_sequence(test_array))
    h_t, c_t, f_t = ts.cal_hidden_state(test)

    # input seeds
    X_train = ts.X_train[random.sample(range(3100), 3000)]

    # test objective NC
    nctoe = NCTestObjectiveEvaluation(r)
    nctoe.model = ts.model
    nctoe.testObjective.layer = layer
    nctoe.testCase = test
    activations_nc = nctoe.get_activations()
    nctoe.testObjective.feature = (np.argwhere(activations_nc >= np.min(activations_nc))).tolist()
    nctoe.testObjective.setOriginalNumOfFeature()

    # test objective CC
    cctoe = CCTestObjectiveEvaluation(r)
    cctoe.model = ts.model
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
    mctoe.model = ts.model
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
    sqtoe.model = ts.model
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


    for test in X_train:
        for i in range(4):
            pred1 = ts.displayInfo(test)
            # get next input test2 from the current input test
            smiles = ts.vect_smile(np.array([test]))
            new_smiles = np.array([sme.randomize_smiles(smiles[0],i)])
            test2 = np.squeeze(ts.smile_vect(new_smiles))

            if not (test2 is None):
                pred2 = ts.displayInfo(test2)
                h_t, c_t, f_t = ts.cal_hidden_state(test2)
                cctoe.hidden = h_t
                ts.updateSample(pred1,pred2,0,True)
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
                writeInfo(r, ts.numSamples, ts.numAdv, ts.perturbations, nctoe.coverage, cctoe.coverage, mctoe.coverage,
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
            if ts.numSamples < int(TestCaseNum) and termin < float(CoverageStop):
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
        if ts.numSamples < int(TestCaseNum) and termin < float(CoverageStop):
            continue
        else:
            break

    print("statistics: \n")
    nctoe.displayCoverage()
    cctoe.displayCoverage()
    mctoe.displayCoverage()
    sqtoe.displayCoverage1()
    sqtoe.displayCoverage2()
    ts.displaySamples()
    ts.displaySuccessRate()



