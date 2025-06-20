import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from sklearn.datasets import make_moons, make_circles, make_classification
    import numpy as np
    import torch as t
    return mo, np


@app.cell
async def _():
    import pandas as pd
    import micropip
    import json
    await micropip.install('altair')
    import altair as alt
    return (alt,)


@app.cell
def _():
    from drawdata import ScatterWidget
    return (ScatterWidget,)


@app.cell
def _(ScatterWidget, mo):
    iwidget = mo.ui.anywidget(ScatterWidget())
    iwidget
    return (iwidget,)


@app.cell
def _(iwidget):
    iwidget.value

    # You can also access the widget's specific properties
    iwidget.data
    iwidget.data_as_polars
    return


@app.cell
def _(mo):
    mo.md(text="# Neural Networks with Marimo \n This is going to be a guide to show some key concepts of neural networks for the non-experts and some cool concepts of marimo for the people already in the field")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Introduction: 

    The goal of this tutorial is dual. On one hand I aim to give a farily simple introduction to neural networks to the non-experts in order to allow them to get familiar with some of the core concepts that are used in the field, and be able to start doing their own research. The other objective is to show to the individuals that are already familiar with these concepts, how marimo notebooks can enhance the clairity of python concepts for learning and push them to switch from jupyter to marimo. 

    The structure of the notebook is the following: in section 2 I will introduce the perceptron model, a linear classifier that will be the basis for the next step; in section 3 we are going through some of the most important activation functions used in the field and I will explain why they are necessary; section 4 will wrap these two concepts together to build a multi-layer perceptron model. Section 5 and 6 are going to be about the loss function and backpropagation to show how the model can learn from the data. Lastly section 7 will show a complete training loop of the model. 

    I have designed this notebook to be interactive so that the user can experiment the different concepts that are present here without the necessity to code them, however, I always hope that the user will try to implement them from scratch as I believe that it is the best learning method, and it is also why I am doing this.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
    # 🧠 Perceptron: The Simplest Neural Unit

    The perceptron is the foundation of neural networks. It's a binary classifier that maps an input vector **x** to an output **y** using a weighted sum and an linear activation function:

    $$
    y = \\text{step}(\\mathbf{w} \\cdot \\mathbf{x} + b)
    $$

    We'll explore this idea by changing weights and visualizing the decision boundary interactively.
    """
    )
    return


@app.cell
def _(np, plt, refresh_matplotlib):
    @refresh_matplotlib
    def plot_perceptron_decision_boundary(X, y, w, b, ax=None, title=None):
        """
        Plots the decision boundary of a perceptron.

        Parameters:
        - X: (n_samples, 2) array of input points
        - y: (n_samples,) array of labels (-1 or 1)
        - w: (2,) array of weights
        - b: scalar bias
        - ax: optional matplotlib axis to plot on
        - title: optional title for the plot
        """
        # Create axis if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))

        # Separate points by class
        class_neg1 = X[y == -1]
        class_pos1 = X[y == 1]

        ax.clear()
        ax.scatter(class_neg1[:, 0], class_neg1[:, 1], color='red', label='Class -1')
        ax.scatter(class_pos1[:, 0], class_pos1[:, 1], color='blue', label='Class +1')

        # Plot decision boundary: w1 * x + w2 * y + b = 0 -> y = -(w1*x + b)/w2
        if w[1] != 0:
            x_vals = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200)
            y_vals = -(w[0] * x_vals + b) / w[1]
            ax.plot(x_vals, y_vals, 'k--', label='Decision Boundary')
        else:
            # Handle vertical line case
            x_val = -b / w[0]
            ax.axvline(x=x_val, color='k', linestyle='--', label='Decision Boundary')

        ax.set_xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
        ax.set_ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")
        ax.legend()
        ax.grid(True)
        if title:
            ax.set_title(title)
    return (plot_perceptron_decision_boundary,)


@app.cell
def _(mo):
    # put ui definitions 
    e = mo.ui.slider(start=0, stop=20, step=1, value=10, label='Number of epochs', show_value=True)
    return (e,)


@app.cell
def _(ImageRefreshWidget, e, np, plot_perceptron_decision_boundary):
    # fig, ax = plt.subplots()
    np.random.seed(42)
    n = 50
    class0 = np.random.randn(n, 2) * 0.05 + np.array([1, 1])
    class1 = np.random.randn(n, 2) * 0.5 + np.array([3, 3])

    X = np.vstack((class0, class1))
    y = np.hstack((np.zeros(n), np.ones(n)))
    y[y == 0] = -1

    # Initialize weights and bias
    w = np.random.randn(2)
    b = 0.0
    lr = 0.1
    widget = ImageRefreshWidget(src=plot_perceptron_decision_boundary(X, y, w, b,  title=f"Epoch {-1}"))
    # Perceptron training loop
    for epoch in range(e.value):
        for i in range(len(X)):
            if y[i] * (np.dot(w, X[i]) + b) <= 0:
                w += lr * y[i] * X[i]
                b += lr * y[i]
                widget.src =plot_perceptron_decision_boundary(X, y, w, b,  title=f"Epoch {epoch + 1}")

        # plt.pause(0.1)

        # plt.show()

    return (widget,)


@app.cell
def _(e, mo, widget):
    mo.hstack([widget, e])
    return


@app.cell
def _(mo):
    mo.md(text="## Notes from Video")
    return


@app.cell
def _():
    import polars as pl
    import random
    return pl, random


@app.cell
def _(alt, mo, np, pl, random):
    data = []
    for j in range(100):
        data.append(random.random()-.5)
        df = pl.DataFrame({"X":range(len(data)), "y":np.cumsum(data)})
        mo.output.replace(alt.Chart(df).mark_line().encode(x="X", y="y"))
    return


@app.cell
def _(mo):
    mo.md(text="How about matplotlib and how can we integrate this?")
    return


@app.cell
def _():
    from mofresh import refresh_matplotlib
    import matplotlib.pylab as plt
    return plt, refresh_matplotlib


@app.cell
def _(np, plt, refresh_matplotlib):
    @refresh_matplotlib
    def cumsum_linechart(data):
        y = np.cumsum(data)
        plt.plot(np.arange(len(y)), y)
    return (cumsum_linechart,)


@app.cell
def _(mo):
    mo.md(text="Here the decoretor takes the img and it converts it inot base64 string that can be plotted by <img> tags in html")
    return


@app.cell
def _():
    from mofresh import ImageRefreshWidget
    import time

    return ImageRefreshWidget, time


@app.cell
def _(mo, random):
    get_state, set_state = mo.state([random.random() - .5])
    return get_state, set_state


@app.cell
def _(cumsum_linechart, get_state, random, set_state, time, widget):
    for _i in range(20):
        set_state(get_state() + [random.random() - .5])
        # this one line over causes the update
        widget.src= cumsum_linechart(get_state())
        time.sleep(0.2)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
