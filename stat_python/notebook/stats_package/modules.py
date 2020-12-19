import numpy as np
import matplotlib.pyplot as plt

linestyles = ["-", "--", ":"]

def E(X, g = lambda x: x) -> float:
    '''
    docstring
    一般化された期待値を求める関数
    _がprefixにつくと使用者が意識しなくて良い変数として扱う意味があった気がする
    '''
    _x_set, f = X
    return np.sum([g(x_k)*f(x_k) for x_k in _x_set])


def V(X, g = lambda x: x) -> float:
    '''
    一般化された分散を求める関数
    '''
    _x_set, f = X
    _mean = E(X, g)
    return np.sum([(g(x_k)-_mean)**2 * f(x_k) for x_k in _x_set])

def check_prob(X) -> None:
    x_set, f=X
    prob=np.array([f(x_k) for x_k in x_set])
    assert np.all(prob >= 0), "負の確率があります"
    prob_sum = np.round(np.sum(prob), 6)
    assert prob_sum == 1, f"確率の和：{prob_sum}になりました"
    print(f"期待値は{E(X):.4}")
    print(f"分散は{V(X):.4}")
    
def plot_prob(X) -> None:
    x_set, f = X
    prob = np.array([f(x_k) for x_k in x_set])
    
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)
    ax.bar(x_set, prob, label="prob")
    ax.vlines(E(X), 0, 1, label="mean")
    ax.set_xticks(np.append(x_set, E(X)))
    ax.set_ylim(0, prob.max()*1.2)
    ax.legend()
    
    plt.show()