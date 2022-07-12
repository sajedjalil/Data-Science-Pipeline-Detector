# Evaluate the mean and std of the submission.
import numpy as np
np.random.seed(71)
import argparse


def load_submission(filename):
    gift_names = ['horse', 'ball', 'bike', 'train', 'coal',
                  'book', 'doll', 'blocks', 'gloves']
    name_to_id = {name[:4]: i for i, name in enumerate(gift_names)}

    with open(filename) as fp:
        lines = fp.readlines()[1:1001]

    bags = []
    for line in lines:
        bag = filter(lambda x: x != "", line.split(" "))
        bags.append(map(lambda x: name_to_id[x[:4]], bag))
    return bags


def get_functions():
    sample_func = [
        lambda size: np.maximum(0, np.random.normal(5, 2, size)),
        lambda size: np.maximum(0, 1 + np.random.normal(1, 0.3, size)),
        lambda size: np.maximum(0, np.random.normal(20, 10, size)),
        lambda size: np.maximum(0, np.random.normal(10, 5, size)),
        lambda size: 47 * np.random.beta(0.5, 0.5, size),
        lambda size: np.random.chisquare(2, size),
        lambda size: np.random.gamma(5, 1, size),
        lambda size: np.random.triangular(5, 10, 20, size),
        lambda size: 3.0 + np.random.rand(size)
        # lambda size: 3.0 + np.array(
        #     [np.random.rand(1)[0] if np.random.rand(1) < 0.3 else np.random.rand(1)[0] \
        #      for _ in xrange(np.prod(size))]).reshape(size)
    ]
    return sample_func


def estimate_value(bags, n_samples):
    total_value = np.zeros(n_samples, dtype=np.float64)
    sample_func = get_functions()

    for bag in bags:
        bag_value = np.zeros(n_samples, dtype=np.float64)
        for gift in bag:
            bag_value += sample_func[gift](n_samples)
        bag_value[bag_value > 50] = 0
        total_value += bag_value

    return total_value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Please uncomment below to use.
    # parser.add_argument("input")
    # parser.add_argument("-n", "--n_samples", default=10**6)
    # parser.add_argument("-th", "--threshold", default=36000)
    # parser.add_argument("-o", "--output_image", default=None)
    args = parser.parse_args()

    # Please comment out below..
    args.input = "../input/sample_submission.csv"
    args.n_samples = 10000
    args.threshold = 10000
    args.output_image = "output_image.png"

    bags = load_submission(args.input)
    estimated_value = estimate_value(bags, args.n_samples)

    print("Mean:", estimated_value.mean())
    print("Std :", estimated_value.std())
    print("Prob:", (estimated_value >= args.threshold).sum() / args.n_samples)
    
    if args.output_image is not None:
        import matplotlib.pyplot as plt
        plt.hist(estimated_value, bins=100, normed=True)
        plt.savefig(args.output_image)
        plt.clf()
