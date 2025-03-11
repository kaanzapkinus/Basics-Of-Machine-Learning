import numpy as np
import pandas as pd

if __name__ == '__main__':
    file = 'first_file.csv'
    df = pd.read_csv(file, header = 0)

    # print(df.head(5))

    '''
    for col in df.columns:
        print(col)
    '''
    X = df[['x']]
    print(X.shape)
    y = df[['y']]

    from sklearn import linear_model
    regression = linear_model.LinearRegression(fit_intercept=True)
    model = regression.fit(X,y)

    print(model.coef_)
    print(model.score(X,y))

    x_pred = np.linspace(0, 40, 200)
    # x_pred [[i] for i in range(0,200,5)]
    x_pred = x_pred.reshape(-1,1)
    y_pred = model.predict(x_pred)

    import matplotlib.pyplot as plt

    plt.style.use('default')
    plt.style.use('ggplot')

    fig, ax = plt.subplots(figsize=(7, 3.5))

    ax.plot(x_pred, y_pred, color='r', label='Regression line', linewidth=4, alpha =0.5)
    ax.scatter(X, y, edgecolor='k', facecolor='blue', alpha=0.7, label='data')
    ax.set_ylabel('y', fontsize=20)
    ax.set_xlabel('x', fontsize=20)
    ax.legend(facecolor='white', fontsize=11)
    ax.text(0.55, 0.15, f'$y = {round(model.coef_[0][0],4)} x - {round(abs(model.intercept_[0]),2)} $', fontsize=17, transform=ax.transAxes)

    fig.tight_layout()
    plt.show()

    file = 'second_file.csv'
    df = pd.read_csv(file, header=0)
    X = df[['x1', 'x2']]
    y = df['y']
    regr = linear_model.LinearRegression()
    regr.fit(X, y)
    print(regr.score(X, y))
    print(regr.coef_)
    print(regr.intercept_)

    plt.show()