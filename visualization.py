import matplotlib.pyplot as plt
import seaborn as sns
import os
import config

class Visualizer:
    def __init__(self):
        sns.set_theme(style=config.STYLE)
        self.palette = config.PALETTE

    def plot_line(self, data, x, y, title, filename, xlabel=None, ylabel=None):
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=data, x=x, y=y, marker='o', palette=self.palette)
        plt.title(title, fontsize=15)
        plt.xlabel(xlabel or x, fontsize=12)
        plt.ylabel(ylabel or y, fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(config.PLOTS_DIR, filename))
        plt.close()

    def plot_bar(self, data, x, y, title, filename, xlabel=None, ylabel=None, horizontal=False):
        plt.figure(figsize=(12, 6))
        if horizontal:
            sns.barplot(data=data, x=y, y=x, palette=self.palette)
        else:
            sns.barplot(data=data, x=x, y=y, palette=self.palette)
        plt.title(title, fontsize=15)
        plt.xlabel(xlabel or x, fontsize=12)
        plt.ylabel(ylabel or y, fontsize=12)
        plt.xticks(rotation=45 if not horizontal else 0)
        plt.tight_layout()
        plt.savefig(os.path.join(config.PLOTS_DIR, filename))
        plt.close()

    def plot_histogram(self, data, x, title, filename, bins=30):
        plt.figure(figsize=(12, 6))
        sns.histplot(data=data, x=x, bins=bins, kde=True, color='teal')
        plt.title(title, fontsize=15)
        plt.tight_layout()
        plt.savefig(os.path.join(config.PLOTS_DIR, filename))
        plt.close()

    def plot_box(self, data, x, y, title, filename):
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=data, x=x, y=y, palette=self.palette)
        plt.title(title, fontsize=15)
        plt.tight_layout()
        plt.savefig(os.path.join(config.PLOTS_DIR, filename))
        plt.close()

    def plot_heatmap(self, data, title, filename):
        plt.figure(figsize=(10, 8))
        sns.heatmap(data, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title(title, fontsize=15)
        plt.tight_layout()
        plt.savefig(os.path.join(config.PLOTS_DIR, filename))
        plt.close()

    def plot_scatter(self, data, x, y, title, filename, hue=None):
        plt.figure(figsize=(12, 6))
        sns.scatterplot(data=data, x=x, y=y, hue=hue, palette=self.palette)
        plt.title(title, fontsize=15)
        plt.tight_layout()
        plt.savefig(os.path.join(config.PLOTS_DIR, filename))
        plt.close()

    def plot_pie(self, data, labels, title, filename):
        plt.figure(figsize=(8, 8))
        plt.pie(data, labels=labels, autopct='%1.1f%%', startangle=140, colors=sns.color_palette(self.palette, len(labels)))
        plt.title(title, fontsize=15)
        plt.tight_layout()
        plt.savefig(os.path.join(config.PLOTS_DIR, filename))
        plt.close()
