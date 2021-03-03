import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
#https://stackoverflow.com/questions/22354094/pythonic-way-of-detecting-outliers-in-one-dimensional-observation-data

def main():
	for num in [10, 50, 100, 1000]:
		# Generate some data
		x = np.random.normal(0, 0.5, num-3)

		# Add three outliers...
		x = np.r_[x, -3, -10, 12]
		plot(x)

	plt.show()

def mad_based_outlier(points, thresh=3.5):
	if len(points.shape) == 1:
		points = points[:,None]
	median = np.median(points, axis=0)
	diff = np.sum((points - median)**2, axis=-1)
	diff = np.sqrt(diff)
	med_abs_deviation = np.median(diff)

	modified_z_score = 0.6745 * diff / med_abs_deviation

	return modified_z_score > thresh

"""def mad_based_outlier(points, thresh=3.5):
	if type(points) is list:
		points = np.asarray(points)
	if len(points.shape) == 1:
		points = points[:, None]
	med = np.median(points, axis=0)
	abs_dev = np.absolute(points - med)
	med_abs_dev = np.median(abs_dev)

	mod_z_score = norm.ppf(0.75) * abs_dev / med_abs_dev
	almost = mod_z_score > thresh
	return [i[0] for i in almost]"""


def percentile_based_outlier(data, threshold=95):
	diff = (100 - threshold) / 2.0
	minval, maxval = np.percentile(data, [diff, 100 - diff])
	return (data < minval) | (data > maxval)

def plot(x):
	fig, axes = plt.subplots(nrows=2)
	for ax, func in zip(axes, [percentile_based_outlier, mad_based_outlier]):
		sns.distplot(x, ax=ax, rug=True, hist=False)
		outliers = x[func(x)]
		ax.plot(outliers, np.zeros_like(outliers), 'ro', clip_on=False)

	kwargs = dict(y=0.95, x=0.05, ha='left', va='top')
	axes[0].set_title('Percentile-based Outliers', **kwargs)
	axes[1].set_title('MAD-based Outliers', **kwargs)
	fig.suptitle('Comparing Outlier Tests with n={}'.format(len(x)), size=14)

if __name__ == "__main__":
	main()
