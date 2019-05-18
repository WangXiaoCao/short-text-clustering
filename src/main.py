from argparse import Namespace
import torch
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import math
import random
import time

train_tokens_file = './data/train_tokens.json'
train_topics_file = './data/train_topics.json'
test_tokens_file = './data/test_tokens.json'
vocab_file = './data/vocab.json'
output_file = 'output.json'


k_docid = 'docid'
k_topic = 'topic'
k_cluster = 'cluster'

##################
# Initialization #
##################

# Set Numpy and PyTorch seeds


def set_seeds(seed, cuda):
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)


argus = Namespace(
    seed=1234,
    cuda=False,
    tfidf_max_features=1000,
)

# Set seeds
set_seeds(seed=argus.seed, cuda=argus.cuda)

# Check CUDA
if not torch.cuda.is_available():
    argus.cuda = False
argus.device = torch.device("cuda" if argus.cuda else "cpu")
print("Using CUDA: {}".format(argus.cuda))

#######################
# Feature Engineering #
#######################


class Vocabulary(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def getTextFromTokenIds(self, tokenIds: list):
        vocab = self.vocab
        vocab_len = len(vocab)

        return ' '.join([vocab[i][0] if i < vocab_len and i >= 0 else '' for i in tokenIds])


def get_tensor_from_tokens(vocab: Vocabulary, tokens_list: list):
    train_texts = []
    for line in tokens_list:
        train_texts.append(vocab.getTextFromTokenIds(line["tokenids"]))

    tfidf = TfidfVectorizer(binary=False, decode_error='ignore',
                            stop_words='english', max_features=argus.tfidf_max_features)

    vc = tfidf.fit_transform(train_texts)
    return torch.from_numpy(vc.todense()).to(argus.device)

#####################
# K-Means Algorithm #
#####################


def pairwise_distance(x, y):
    x = x.unsqueeze(dim=1)
    y = y.unsqueeze(dim=0)

    dis = (x-y)**2.0
    dis = dis.sum(dim=-1).squeeze()
    return dis


def k_means_pick_centers(inputs, n_clusters):
    item_count = len(inputs)
    center_indexes = random.sample(range(item_count), n_clusters)

    return torch.stack([inputs[i] for i in center_indexes])


def k_means_center_shift(centers_x, centers_y):
    return torch.sum(torch.sqrt(torch.sum((centers_x - centers_y) ** 2, dim=1)))


def k_means(inputs, n_clusters, max_cycles=float('inf'), tol=1e-4):
    since = time.time()

    cycle_no = 0

    centers = k_means_pick_centers(inputs, n_clusters)

    while True:
        dis = pairwise_distance(inputs, centers)

        choice_cluster = torch.argmin(dis, dim=1)
        centers_pre = centers.clone()

        for index in range(n_clusters):
            selected = torch.nonzero(choice_cluster == index).squeeze()
            selected = torch.index_select(inputs, 0, selected)
            centers[index] = selected.mean(dim=0)

        center_shift = k_means_center_shift(centers, centers_pre)

        if math.isnan(center_shift):
            raise Exception('Computing Error: invalid `nan` value, please restart')

        print('Cycle {}{} Center Shift: {:.4f}'.format(cycle_no, "/" +
                                                       str(max_cycles) if max_cycles < float('inf') else "", center_shift))

        if center_shift ** 2 < tol:
            break

        cycle_no += 1
        if cycle_no >= max_cycles:
            break

    time_elapsed = time.time() - since
    print('========================================')
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    return choice_cluster, centers

#######################
# Execution Procedure #
#######################


def execute_cluster(tokens_list):
    """
    输入一组文档列表，返回每条文档对应的聚类编号
    :param tokens_list: list，每个元素为dict，dict中键值含义如下：
        'docid': int，文档标识符
        'tokenids': list，每个元素为int，单词标识符
    :return: list，每个元素为dict，dict中键值含义如下：
        'docid': int，文档标识符
        'cluster': int，标识该文档的聚类编号
    """
    print(type(tokens_list), type(tokens_list[0]), list(
        tokens_list[0].items()))    # 仅用于验证数据格式

    vocab = Vocabulary(load_array(vocab_file))

    Input = get_tensor_from_tokens(vocab, tokens_list)

    labels, centers = k_means(Input, n_clusters=63 if args.test else 65 , tol=1e-4)

    clusters_list = []

    for i, label in enumerate(labels):
        item = {
            "docid": tokens_list[i]["docid"],
            "cluster": label.item(),
        }

        clusters_list.append(item)

    return clusters_list


""" 以下内容修改无效 """


def calculate_nmi(topics_list, clusters_list):
    id2topic = dict([(d[k_docid], d[k_topic]) for d in topics_list])
    id2cluster = dict([(d[k_docid], d[k_cluster]) for d in clusters_list])
    common_idset = set(id2topic.keys()).intersection(id2cluster.keys())
    if not len(common_idset) == len(topics_list) == len(clusters_list):
        print(len(common_idset), len(topics_list), len(clusters_list))
        print('length inconsistent, result invalid')
        return 0
    else:
        topic_cluster = [(id2topic[docid], id2cluster[docid])
                         for docid in common_idset]
        y_topic, y_cluster = list(zip(*topic_cluster))
        nmi = metrics.normalized_mutual_info_score(y_topic, y_cluster)
        print('nmi:{}'.format(round(nmi, 4)))
        return nmi


def dump_array(file, array):
    lines = [json.dumps(item, sort_keys=True) + '\n' for item in array]
    with open(file, 'w') as fp:
        fp.writelines(lines)


def load_array(file):
    with open(file, 'r') as fp:
        array = [json.loads(line.strip()) for line in fp.readlines()]
    return array


def clean_clusters_list(clusters_list):
    return [dict([(k, int(d[k])) for k in [k_docid, k_cluster]]) for d in clusters_list]


def evaluate_train_result():
    train_tokens_list = load_array(train_tokens_file)
    train_clusters_list = execute_cluster(train_tokens_list)
    train_topics_list = load_array(train_topics_file)
    calculate_nmi(train_topics_list, train_clusters_list)


def generate_test_result():
    test_tokens_list = load_array(test_tokens_file)
    test_clusters_list = execute_cluster(test_tokens_list)
    test_clusters_list = clean_clusters_list(test_clusters_list)
    dump_array(output_file, test_clusters_list)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    args = parser.parse_args()
    if args.train:
        evaluate_train_result()
    elif args.test:
        generate_test_result()
