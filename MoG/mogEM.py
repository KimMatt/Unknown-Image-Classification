from kmeans import *
import sys
import matplotlib.pyplot as plt
plt.ion()

if sys.version_info.major == 3:
    raw_input = input


def mogEM(x, K, iters, randConst=1, minVary=0):
    """
    Fits a Mixture of K Diagonal Gaussians on x.

    Inputs:
      x: data with one data vector in each column.
      K: Number of Gaussians.
      iters: Number of EM iterations.
      randConst: scalar to control the initial mixing coefficients
      minVary: minimum variance of each Gaussian.

    Returns:
      p: probabilities of clusters (or mixing coefficients).
      mu: mean of the clusters, one in each column.
      vary: variances for the cth cluster, one in each column.
      logLikelihood: log-likelihood of data after every iteration.
    """
    print(x.shape)
    N, T = x.shape

    # Initialize the parameters
    p = randConst + np.random.rand(K, 1)
    p = p / np.sum(p)   # mixing coefficients
    mn = np.mean(x, axis=1).reshape(-1, 1)
    vr = np.var(x, axis=1).reshape(-1, 1)

    # Question 4.3: change the initializaiton with Kmeans here

    #--------------------  Add your code here --------------------
    #mu = mn + np.random.randn(N, K) * (np.sqrt(vr) / randConst)

    mu = KMeans(x,K,5)



    #------------------------------------------------------------
    vary = vr * np.ones((1, K)) * 2
    vary = (vary >= minVary) * vary + (vary < minVary) * minVary

    logLikelihood = np.zeros((iters, 1))

    # Do iters iterations of EM
    for i in xrange(iters):
        # Do the E step
        respTot = np.zeros((K, 1))
        respX = np.zeros((N, K))
        respDist = np.zeros((N, K))
        ivary = 1 / vary
        logNorm = np.log(p) - 0.5 * N * np.log(2 * np.pi) - \
            0.5 * np.sum(np.log(vary), axis=0).reshape(-1, 1)
        logPcAndx = np.zeros((K, T))
        for k in xrange(K):
            dis = (x - mu[:, k].reshape(-1, 1))**2
            logPcAndx[k, :] = logNorm[k] - 0.5 * \
                np.sum(ivary[:, k].reshape(-1, 1) * dis, axis=0)

        mx = np.max(logPcAndx, axis=0).reshape(1, -1)
        PcAndx = np.exp(logPcAndx - mx)
        Px = np.sum(PcAndx, axis=0).reshape(1, -1)
        PcGivenx = PcAndx / Px
        logLikelihood[i] = np.sum(np.log(Px) + mx)

        print 'Iter %d logLikelihood %.5f' % (i + 1, logLikelihood[i])

        # Plot log likelihood of data
        plt.figure(0)
        plt.clf()
        plt.plot(np.arange(i), logLikelihood[:i], 'r-')
        plt.title('Log-likelihood of data versus # iterations of EM')
        plt.xlabel('Iterations of EM')
        plt.ylabel('Log-likelihood')
        plt.draw()

        # Do the M step
        # update mixing coefficients
        respTot = np.mean(PcGivenx, axis=1).reshape(-1, 1)
        p = respTot

        # update mean
        respX = np.zeros((N, K))
        for k in xrange(K):
            respX[:, k] = np.mean(x * PcGivenx[k, :].reshape(1, -1), axis=1)

        mu = respX / respTot.T

        # update variance
        respDist = np.zeros((N, K))
        for k in xrange(K):
            respDist[:, k] = np.mean(
                (x - mu[:, k].reshape(-1, 1))**2 * PcGivenx[k, :].reshape(1, -1), axis=1)

        vary = respDist / respTot.T
        vary = (vary >= minVary) * vary + (vary < minVary) * minVary

    return p, mu, vary, logLikelihood


def mogLogLikelihood(p, mu, vary, x):
    """ Computes log-likelihood of data under the specified MoG model

    Inputs:
      x: data with one data vector in each column.
      p: probabilities of clusters.
      mu: mean of the clusters, one in each column.
      vary: variances for the cth cluster, one in each column.

    Returns:
      logLikelihood: log-likelihood of data after every iteration.
    """
    print(x.shape)
    K = p.shape[0]
    N, T = x.shape
    ivary = 1 / vary
    logLikelihood = np.zeros(T)
    for t in xrange(T):
        # Compute log P(c)p(x|c) and then log p(x)
        logPcAndx = np.log(p) - 0.5 * N * np.log(2 * np.pi) \
            - 0.5 * np.sum(np.log(vary), axis=0).reshape(-1, 1) \
            - 0.5 * \
            np.sum(ivary * (x[:, t].reshape(-1, 1) - mu)
                   ** 2, axis=0).reshape(-1, 1)

        mx = np.max(logPcAndx, axis=0)
        logLikelihood[t] = np.log(np.sum(np.exp(logPcAndx - mx))) + mx

    return logLikelihood

def findrandConst():
    K = 5
    iters = 10
    minVary = 0.01
    randConst = 1.0

    # load data
    inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = LoadData(
        '../regular_fruits.npz')

    randConstants = [.25,.5,.75,1.0,1.5,2.0,2.5,3,3.5,4,5,6,7,8]
    likelihoods = []

    for i in randConstants:
        p,mu,vary,logLikelihood = mogEM(inputs_train,K,iters,i,minVary)
        likelihoods.append(logLikelihood[-1][0])


    plt.clf()
    plt.plot(randConstants, likelihoods, 'b', label='randConstants vs likelihoods')
    plt.xlabel('randConstants')
    plt.ylabel('likelihoods')
    plt.legend()
    plt.draw()
    plt.show()


def getAccuracy(likelihoods, priors, y):
    '''
    Calculate the accuracy on a dataset
    '''
    likelihoods = np.array(likelihoods).T + priors
    err = 0
    for index, each in enumerate(likelihoods):
        if np.argmax(each) != y[index]:
            err += 1

    return 1.0 - float(err) / float(len(y))

def getAccuracyThresh(likelihoods, priors, y, threshold):
    '''
    Calculate the accuracy on a dataset
    '''
    likelihoods = np.array(likelihoods).T + priors
    acc = 0
    for index, each in enumerate(likelihoods):
        if np.argmax(each) == y[index] and each[np.argmax(each)] > threshold:
            acc += 1

    return float(acc) / float(len(y))

def getClassified(likelihoods, priors, threshold):
    '''
    Calculate the amount of classified examples
    '''
    likelihoods = np.array(likelihoods).T + priors
    classified = 0
    for index, each in enumerate(likelihoods):
        if each[np.argmax(each)] > threshold:
            classified +=1

    return float(classified) / float(len(likelihoods))

def classifyFruits():
    iters = 10
    minVary = .01
    randConst = 1.0

    numComponents = 45
    classifications = []
    thresholds = [-1600,-900,-800,-700,-600,-500,-400,-300,-200,-100,0, 50, 100, 200, 300, 400, 500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600]
    accuracies = []

    # extract data of class 1-Anger, 4-Happy
    dataQ4 = LoadDataQ4('../regular_fruits.npz')
    dataUnseen = np.load('../test.npz')['inputs']
    dushape = dataUnseen.shape
    dataUnseen = np.reshape(dataUnseen, (dushape[0], dushape[1] * dushape[2])).T
    # images
    x_train_orange = dataQ4['x_train_orange']
    x_train_apple = dataQ4['x_train_apple']
    x_train_banana = dataQ4['x_train_banana']

    x_train = np.concatenate([x_train_orange, x_train_apple, x_train_banana], axis=1)
    x_valid = np.concatenate(
        [dataQ4['x_valid_orange'], dataQ4['x_valid_apple'], dataQ4['x_valid_banana']], axis=1)
    x_test = np.concatenate(
        [dataQ4['x_test_orange'], dataQ4['x_test_apple'], dataQ4['x_test_banana']], axis=1)

    # label
    y_train = np.concatenate(
        [dataQ4['y_train_orange'], dataQ4['y_train_apple'], dataQ4['y_train_banana']])
    y_valid = np.concatenate(
        [dataQ4['y_valid_orange'], dataQ4['y_valid_apple'], dataQ4['y_valid_banana']])
    y_test = np.concatenate([dataQ4['y_test_orange'], dataQ4['y_test_apple'], dataQ4['y_test_banana']])

    # Hints: this is p(d), use it based on Bayes Theorem
    num_orange_train = x_train_orange.shape[1]
    num_apple_train = x_train_apple.shape[1]
    num_banana_train = x_train_banana.shape[1]
    log_likelihood_class = np.log(
        [num_orange_train, num_apple_train, num_banana_train]) - np.log(num_orange_train + num_apple_train + num_banana_train)

    K = numComponents

    # Train a MoG model with K components
    # Hints: using (x_train_anger, x_train_happy) train 2 MoGs
    #-------------------- Add your code here ------------------------------

    p_o, mu_o, vary_o, logLikelihood_o = mogEM(x_train_orange, K, iters, randConst, minVary)
    p_a, mu_a, vary_a, logLikelihood_a = mogEM(x_train_apple, K, iters, randConst, minVary)
    p_b, mu_b, vary_b, logLikelihood_b = mogEM(x_train_orange, K, iters, randConst, minVary)
    # Compute the probability P(d|x), classify examples, and compute error rate
    # Hints: using (x_train, y_train), (x_valid, y_valid), (x_test, y_test)
    # to compute error rates, you may want to use mogLogLikelihood function
    #-------------------- Add your code here ------------------------------



    classlikelihood_train_o = mogLogLikelihood(p_o,mu_o,vary_o,x_train)
    classlikelihood_unseen_o = mogLogLikelihood(p_o, mu_o, vary_o, dataUnseen)
    classlikelihood_train_a = mogLogLikelihood(p_a,mu_a,vary_a,x_train)
    classlikelihood_train_b = mogLogLikelihood(p_b,mu_b,vary_b,x_train)

    train_acc = getAccuracy([classlikelihood_train_o, classlikelihood_train_a, classlikelihood_train_b], log_likelihood_class, y_train)
    
    classlikelihood_valid_o = mogLogLikelihood(p_o,mu_o,vary_o,x_valid)
    classlikelihood_valid_a = mogLogLikelihood(p_a,mu_a,vary_a,x_valid)
    classlikelihood_valid_b = mogLogLikelihood(p_b,mu_b,vary_b,x_valid)

    valid_acc = getAccuracy([classlikelihood_valid_o, classlikelihood_valid_a, classlikelihood_valid_b], log_likelihood_class, y_valid)

    classlikelihood_test_o = mogLogLikelihood(p_o,mu_o,vary_o,x_test)
    classlikelihood_test_a = mogLogLikelihood(p_a,mu_a,vary_a,x_test)
    classlikelihood_test_b = mogLogLikelihood(p_b,mu_b,vary_b,x_test)

    test_acc = getAccuracy([classlikelihood_test_o, classlikelihood_test_a, classlikelihood_test_b], log_likelihood_class, y_test)

    print("No threshold: ")
    print("train_acc: " + str(train_acc))
    print("valid_acc: " + str(valid_acc))
    print("test_acc: " + str(test_acc))

    classlikelihood_unseen_o = mogLogLikelihood(p_o, mu_o, vary_o, dataUnseen)
    classlikelihood_unseen_a = mogLogLikelihood(p_a, mu_a, vary_a, dataUnseen)
    classlikelihood_unseen_b = mogLogLikelihood(p_b, mu_b, vary_b, dataUnseen)

    for threshold in thresholds:
        thresh_acc = getAccuracyThresh([classlikelihood_test_o, classlikelihood_test_a, classlikelihood_test_b], log_likelihood_class, y_test, threshold)
        accuracies.append(thresh_acc)
        classified = getClassified([classlikelihood_unseen_o, classlikelihood_unseen_a, classlikelihood_unseen_b], log_likelihood_class, threshold)
        classifications.append(classified)

    # Plot the error rate
    plt.figure(0)
    plt.clf()
    #-------------------- Add your code here --------------------------------

    print('-----------_classification rates_-----------------')
    print(classifications)
    print('-----------_likelihoods_-----------------')
    print(classlikelihood_unseen_b)
    print('---------------')
    print(accuracies)

    print("Classified at threshold 0?:")

    testclassified = getClassified([classlikelihood_test_o, classlikelihood_test_b, classlikelihood_test_a], log_likelihood_class, 0)
    validclassified = getClassified([classlikelihood_valid_o, classlikelihood_valid_b, classlikelihood_valid_a], log_likelihood_class, 0)
    trainingclassified = getClassified([classlikelihood_train_o, classlikelihood_train_b, classlikelihood_train_a], log_likelihood_class, 0)
  
    print('test: ' + str(testclassified))
    print('valid: ' + str(validclassified))
    print('training: ' + str(trainingclassified))

    plt.plot(thresholds, classifications, 'r', label='Classifications')
    plt.plot(thresholds, accuracies, 'g', label='Testing Accuracies')
    plt.xlabel('Thresholds')
    plt.ylabel('Percentage')
    plt.legend()
    plt.draw()
    plt.pause(0.0001)

if __name__ == '__main__':

    #findrandConst()
    classifyFruits()

    raw_input('Press Enter to continue.')
