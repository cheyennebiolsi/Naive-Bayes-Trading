# (2016) Developed by Cheyenne Biolsi for Bayesquare Foundation Inc.

import numpy as np
from sklearn.naive_bayes import GaussianNB
import csv
import argparse
import os
import matplotlib.pyplot as plt
import pandas
from datetime import date, datetime
from time import strftime
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from asset import Asset
from dateutil.parser import parse

##
# Takes in a comma separated txt file and
# returns a list of float lists, where each
# float list represents a row in the original
# file.
#
def loadCsv(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=",")
        dataset = list(reader)
    return dataset

##
# Returns n random positive floats that sum
# to 1.
#
def rand_weights(n):
    k = np.random.rand(n)
    return k / sum(k)

##
# Returns the daily return for the dth row
# of a dataset, where close represents
# which column of the dataset contains the
# close prices
#
def calculate_return(dataset, close, d):
    try:
        return(100*np.log(float(dataset[d][close])/float(dataset[d-1][close])))
    except ValueError:
        print("Not enough data in dataset")

##
# Given the asset weights, historical returns,
# and expected returns, generates the risk
# and return for a portfolio with those weights
#
def risksAndReturns(w, volatilities, expectations):
    w = np.asmatrix(w)
    p = np.asmatrix(expectations)
    C = np.asmatrix(np.diag(volatilities))
    
    mu = (w * p.T).item(0)
    sigma = np.sqrt(w * C * w.T).item(0)
    return mu, sigma

##
# Returns the weights of the tangency portfolio
#
def tangencyPortfolio(volatilities, expectations):
    n = len(volatilities)
    m = np.asmatrix(expectations).T
    e = np.asmatrix(np.ones((n,1)))
    cov = np.asmatrix(np.diag(volatilities))
    invCov = cov.I
    
    w = (invCov*m)/(e.T * invCov * m)
    return(w)

##
# Implements Markowitz solution for portfolio problem
# using closed-form solution
#
def markowitz(volatilities, expectations, baseline = 0.01):
    n = len(volatilities)
    m = np.asmatrix(expectations).T
    e = np.asmatrix(np.ones((n,1)))
    cov = np.asmatrix(np.diag(volatilities))
    invCov = cov.I

    delta = (m.T * invCov * m) * (e.T*invCov*e) - ((m.T * invCov * e)*(m.T * invCov * e))
    alpha = (1/delta) * (baseline * (m.T*invCov*e) * (e.T*invCov * e) - ((m.T * invCov * e)*(m.T * invCov * e)))
    minvar = invCov*e * (1/ (e.T*invCov*e))
    mk = invCov*m * (1/ (e.T*invCov*m))
    w = np.asarray((1-alpha.item(0))*minvar + alpha.item(0)*mk)
    w = w.reshape(1,-1).flatten().tolist()
    return(w)

##
# Find index of a column with a specific name
#
def findColumn(dataset, column_name):
    try:
        column_names = dataset[0]
        return(next(i for i,v in enumerate(column_names) if v.lower().strip() == column_name))
    except:
        print("Impromperly named column")

##
# Find the index of the starting date in the dataset
#
def findStartIndex(dataset, start_date):
    date_column = findColumn(dataset, 'date')
    try:
        start = parse(start_date)
    except:
        print("Invalid start date entered")
    for i in range(1, len(dataset)):
        if start <= parse(dataset[i][date_column]):
            return i
    return -1
        

##
# Returns a tuple of arrays X and Y, where Y is a binary variable
# representing whether an asset in file f on day d experienced a positive
# or negative return, and X where each row d contains the historical
# returns of the previous 1 to p days.
#
# Additionally, it allows for other columns of data to be
# represented in X.  For each column c in columns for a day d,
# the values in c will be represented for the previous 1 to p days.
#
def prepareDataset(f, columns, prior, trainingDays, tradingDays, start_date):
    dataset = loadCsv(f)
    close = findColumn(dataset, 'close')
    start_index = findStartIndex(dataset, start_date)
    X = []
    Y = []
    for i in range(start_index - trainingDays, start_index + tradingDays + 1):
        x = []
        for p in range(1, prior + 1):
            x.append(calculate_return(dataset, close, i-p))
        if columns!=None:
            for c in columns:
                for p in range(1, prior + 1):
                    x.append(dataset[i-p][findColumn(dataset, c)])
        if calculate_return(dataset, close, i)>0:
            Y.append(0)
        else:
            Y.append(1)
        x.append(parse(dataset[i][0]))
        X.append(x)
    return(np.array(X), np.array(Y))

##
# Creates a PDF of plots
#
def createPDF(figs):
    with PdfPages("nb_plots.pdf") as pdf:
        for fig in figs:
            pdf.savefig(fig)

##
# Creates either a classifier fitted using the asset's history
#
def createClassifier(asset):
    classifier = GaussianNB()
    classifier.fit(asset.assethistory, asset.classvalues)
    return classifier

##
# Returns expectations and markowitz optimized weights for each asset
# for each trading day.
#
def simulateDays(tradingDays, assets, tradingData, tradingClasses, plot):
    dates = []
    figs = []
    weights = []
    expectationsMatrix = []
    for t in range(tradingDays):
        expectations = []
        volatilities = []
        for i in range(len(assets)):
            asset = assets[i]
            testSet = tradingData[i][t][:-1].reshape(1,-1) #ignore last element (the date)
            classifier = createClassifier(asset)
            e = getExpectedReturn(classifier, testSet, asset)
            v = asset.update_volatility()
            asset.update(testSet, tradingClasses[i][t])
            expectations.append(e)
            volatilities.append(v)
        date = tradingData[0][t][-1]
        dates.append(date)
        
        tanWeights = tangencyPortfolio(volatilities, expectations).reshape(1,-1).tolist()[0]
        mu, _ = risksAndReturns(tanWeights, volatilities, expectations)
        
        if plot == 1:
            figs.append(plotRisksReturns(tanWeights, expectations, volatilities, date))
        
        # If the expected return of the tangency portfolio is below the risk-free,
        # short all assets
        if mu < 0:
            tanWeights = -1*np.array(tanWeights)
        
        weights.append(tanWeights)
        expectationsMatrix.append(expectations)       
    return(expectationsMatrix, weights, dates, figs)

##
# Returns the expected movement for an asset based on
# a classifier and previous average movement
#
def getExpectedReturn(model, testSet, asset):
    (averageIncrease, averageDecrease) = asset.averages()
    probability = model.predict_proba(testSet)
    positiveProb = probability[0][0]
    negativeProb = probability[0][1]
    movement = positiveProb*averageIncrease + negativeProb*averageDecrease
    return(movement)           

##
# Returns a list of asset objects for each asset in a database
#
def create_assets(database, classvalues, names, trainingDays):
    assets = []
    tradingData = []
    tradingClasses = []
    for i in range(len(database)):
        assethistory = database[i][:trainingDays, :-1]
        asset = Asset(names[i][2:-4], assethistory = assethistory, classvalues = classvalues[i][:trainingDays])
        tradingData.append(database[i][trainingDays:])
        tradingClasses.append(classvalues[i][trainingDays:])
        asset.recount_averages()
        assets.append(asset)
        
    return assets, tradingData, tradingClasses

##
# Returns the ideal weights, and expected and actual portfolio returns
# for a directory of assets over the course of tradingDays days.
#
def simulate(directory, columns, prior, trainingDays, tradingDays, start_date, plot):
    database = []
    classvalues = []
    names = []
    for root, dirs, filenames in os.walk(directory):
        for f in filenames[2:6]:
            X, Y = prepareDataset(directory + '/' + f, columns, prior, trainingDays, tradingDays, start_date)
            database.append(X)
            classvalues.append(Y)
            names.append(f)
            
    assets, tradingData, tradingClasses = create_assets(database, classvalues, names, trainingDays)
    e, w, d, figs = simulateDays(tradingDays, assets, tradingData, tradingClasses, plot)
    a = getActualReturns(database, trainingDays, tradingDays)
    portfolioReturns = getPortfolioReturns(w, a)
    theorizedReturns = getPortfolioReturns(w, e)
    
    w = np.asarray(w).T
    printTable(w, names, d)
    
    figs.append(plotTrades(w, names, portfolioReturns, theorizedReturns, d))
    return(w, names, portfolioReturns, theorizedReturns, figs, d)

##
# Calculates the returns for each asset for each day
#
def getActualReturns(database, trainingDays, tradingDays):
    returns = []
    for t in range(tradingDays):
        rets = []
        for i in range(len(database)):
            dataset=database[i]
            rets.append(dataset[trainingDays+t+1, 0])
        returns.append(rets)
    return returns

##
# Calculates expected portfolio returns based on
# weights and expected asset return
#
def getPortfolioReturns(weights, returns):
    portfolio = []
    for i in range(len(weights)):
        dayWeight = np.asarray(weights[i])
        dayReturns = np.asarray(returns[i]).T
        portfolio.append(np.dot(dayWeight, dayReturns))
    return portfolio

##
# Calculates the efficient frontier portfolios' returns and risks
# for a range of returns based on the tangency portfolio's
# returns
#
def calculateFrontier(tanRet, volatilities, expectations):
    eM = []; eV = []
    endWeights = markowitz(volatilities, expectations, baseline=1.8*tanRet)
    endM, endV = risksAndReturns(np.asarray([endWeights]), volatilities, expectations)
    
    # Calculate optimal weights of frontier and get risks and returns
    mus = np.linspace(-5*tanRet, 5*tanRet, 500)
    frontier = []
    for mu in mus:
        w = markowitz(volatilities, expectations, baseline=mu)
        frontier.append(w)
    
    for w in frontier:
        m, v = risksAndReturns(np.asarray([w]), volatilities, expectations)
        #Create an arc that is about symmetrical about the x axis
        if v <= endV:
            eM.append(m)
            eV.append(v)
    return eM, eV         
    
##
# Plots the efficient frontier, tangency portfolio and tangent line through it,
# and random portfolios
#
def plotRisksReturns(tanWeights, expectations, volatilities, t):
    M = []; V = []
    n = len(tanWeights)
    tanRet, tanRisk = risksAndReturns(tanWeights, volatilities, expectations)
    
    eM, eV = calculateFrontier(tanRet, volatilities, expectations)
    
    # Generate 500 random portfolios
    for i in range(500):
        w = np.asmatrix(rand_weights(n))
        m, v = risksAndReturns(w, volatilities, expectations)
        M.append(m)
        V.append(v)
    
    #create function to add percentage signs    
    def to_percent(y, pos=0):
        s = str(y)
        return s + '%'
    
    formatter = FuncFormatter(to_percent)
    
    # Draw plot
    fig, ax1 = plt.subplots(1, figsize=(16,9))
    ax1.set_xlabel("Standard Deviation")
    ax1.set_ylabel("Expected Return")
    title = datetime.strftime(t, "Efficient frontier for %b %d, %Y")
    fig.suptitle(title, fontsize = 14, fontweight = 'bold')
    ax1.set_title('Using Naive-Bayes classifier', fontsize = 10)
    
    #Create tangent line    
    xp = np.linspace(0.0, 2*tanRisk, 200)
    tan = np.polyfit([0.0, tanRisk], [0.0, tanRet], 1)
    p2 = np.poly1d(tan)
    ax1.plot(xp, p2(xp), color = "blue")
    
    ax1.plot(eV, eM, color = "red", label = "efficient frontier")
    ax1.plot(V, M, 'o', color = "green", label = "random portfolios")
    ax1.plot([tanRisk], [tanRet], 'o', markersize = 7, color = "cyan", label = "optimal weights")
    
    # Draw arrow for sharpe ratio
    sharpe = "Sharpe: " + format((tanRet/tanRisk), '.2f')
    ax1.annotate(sharpe, xy = (tanRisk, tanRet), xytext = (-20, 20),
                     textcoords = 'offset points', ha = 'right', va = 'bottom',
                     bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
                     arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
    ax1.yaxis.set_major_formatter(formatter)
    ax1.legend(loc = 4, numpoints=1)
    plt.grid()
    plt.close()
    return fig

##
# Returns the accuracy of actual and predicted returns.
#
def getAccuracy(actual, expected):
    n = len(actual)
    positiveCount = 0
    for i in range(n):
        if (actual[i] > 0 and expected[i] > 0) or (actual[i] <= 0 and expected[i] <= 0):
            positiveCount += 1
    return(positiveCount/n)
##
# Calculates the Sharpe ratio
#
def calculateSharpe(returns, riskfree=0):
    returns = np.asarray(returns).reshape(1, len(returns))
    mean = np.mean(returns, axis=1)
    std = np.sqrt(np.var(returns, axis=1))
    s = (mean - riskfree)/std
    return(s)
##
# Plots the ideal weights of a portfolio over several
# trading days.
#
def plotTrades(weights, names, portfolio, expected, dates):
    x = []
    fig, (ax1, ax2) = plt.subplots(2, figsize=(16,9))
    
    title = "Results using Naive Bayes classifier"
    fig.suptitle(title, fontsize = 18, fontweight = 'bold')

    ax1.plot(dates, expected, label = "Expected returns")
    ax1.plot(dates, portfolio, label = "Actual returns")
    ax1.set_ylabel("returns")
    
    start = dates[0].strftime('%b %d, %Y')
    end = dates[-1].strftime('%b %d, %Y')
    
    ax1.set_title("Actual vs expected returns between {} and {}".format(start, end))
    ax1.legend(loc = 0)
    ax1.grid()
    ax1.fmt_xdata = mdates.DateFormatter('%m-%d-%Y')
    
    for i in range(weights.shape[0]):
        ax2.plot(dates, weights[i,:], label = names[i][2:-4])
    ax2.fmt_xdata = mdates.DateFormatter('%m-%d-%Y')
    ax2.legend(loc = 0)
    ax2.set_ylabel("weights")
    ax2.set_title("Optimal portfolio weights between {} and {}".format(start, end))
    ax2.grid()
    plt.close()
    
    return fig

##
# Creates a .csv file of the markowitz optimized weights for each day
# of trading
#
def printTable(weights, names, dates):
    title = "nb_weights.csv"
    
    names = [name[2:-4] for name in names]
    df = pandas.DataFrame(weights, names, dates)
    df.to_csv(title)
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", help = "select which directory to open")
    parser.add_argument("-t", type = int, help = "how many trading days to simulate", default = 100)
    parser.add_argument("-p", "--plotting", type = int, help = "1 to plot efficient frontier, 0 not to", default = 0)
    parser.add_argument("-d", type = int, help = "how long is the training window for the classifier", default = 60)
    parser.add_argument("-s", type = str, help = "start date", default = "20000101")
    parser.add_argument('-a', '--arg', nargs='+', type=str, help = "names of additional columns of data, lowercase")
    parser.add_argument('-y', type = int, help = "data from how many of the previous days to be used as predictors, " +
                        "defaults to yesterday's", default = 1)
    args = parser.parse_args()
    indir = args.directory
    tradingDays = args.t
    plot = args.plotting
    trainingDays = args.d
    start_date = args.s
    prior = args.y
    columns = args.arg
   
    w, names, portfolio, theorized, figs, dates = simulate(indir, columns, prior, trainingDays,
                                                           tradingDays, start_date, plot)
    
    print("Actual Sharpe Ratio: {}".format(calculateSharpe(portfolio).item(0)))
    print("Expected Sharpe Ratio: {}".format(calculateSharpe(theorized).item(0)))
    createPDF(figs)
    
main()
